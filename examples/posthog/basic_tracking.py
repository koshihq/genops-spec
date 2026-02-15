#!/usr/bin/env python3
"""
PostHog Basic Product Analytics Tracking Example

This example demonstrates basic PostHog event tracking with GenOps governance,
including cost attribution, team management, and budget enforcement.

Usage:
    python basic_tracking.py

Prerequisites:
    pip install genops[posthog]
    export POSTHOG_API_KEY="phc_your-project-api-key"
    export GENOPS_TEAM="your-team-name"
    export GENOPS_PROJECT="your-project-name"
"""

import os
import time


def main():
    """Run basic PostHog analytics tracking with GenOps governance."""
    print("🚀 PostHog + GenOps Basic Product Analytics Example")
    print("=" * 60)

    # Prerequisites check
    print("\n📋 Prerequisites Check:")
    prerequisites = [
        ("GenOps installed", "genops"),
        ("PostHog SDK available", "posthog"),
        ("POSTHOG_API_KEY configured", lambda: bool(os.getenv("POSTHOG_API_KEY"))),
        ("GENOPS_TEAM configured", lambda: bool(os.getenv("GENOPS_TEAM"))),
    ]

    for desc, check in prerequisites:
        try:
            if callable(check):
                result = check()
            else:
                __import__(check)
                result = True
            print(f"  ✅ {desc}")
        except (ImportError, Exception):
            print(f"  ❌ {desc}")
            if desc.startswith("GenOps"):
                print("     Fix: pip install genops[posthog]")
            elif "API_KEY" in desc:
                print("     Fix: export POSTHOG_API_KEY='phc_your_api_key'")
            elif "TEAM" in desc:
                print("     Fix: export GENOPS_TEAM='your-team-name'")

    # Initialize GenOps PostHog adapter
    print("\n🎯 Initializing PostHog analytics with governance...")

    try:
        from genops.providers.posthog import GenOpsPostHogAdapter

        # Configuration from environment
        adapter = GenOpsPostHogAdapter(
            posthog_api_key=os.getenv("POSTHOG_API_KEY"),
            team=os.getenv("GENOPS_TEAM", "basic-tracking-team"),
            project=os.getenv("GENOPS_PROJECT", "product-analytics-demo"),
            environment=os.getenv("GENOPS_ENVIRONMENT", "development"),
            daily_budget_limit=float(os.getenv("GENOPS_DAILY_BUDGET_LIMIT", "50.0")),
            enable_governance=True,
            enable_cost_alerts=True,
            governance_policy=os.getenv("GENOPS_GOVERNANCE_POLICY", "advisory"),
        )

        print(
            f"✅ Adapter initialized for team '{adapter.team}', project '{adapter.project}'"
        )

    except Exception as e:
        print(f"❌ Failed to initialize adapter: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Run setup validation: python setup_validation.py")
        print("2. Check your PostHog API key configuration")
        return

    # Demo analytics session with various events
    print("\n📊 Starting analytics session with governance tracking...")

    try:
        with adapter.track_analytics_session(
            session_name="user_onboarding_flow",
            customer_id=os.getenv("DEMO_CUSTOMER_ID", "demo_customer_123"),
            cost_center=os.getenv("DEMO_COST_CENTER", "product"),
            feature="onboarding",
            experiment="signup_optimization_v2",
        ) as session:
            print(
                f"📈 Session started: {session.session_name} ({session.session_id[:8]}...)"
            )

            # Simulate user onboarding events
            onboarding_events = [
                (
                    "landing_page_viewed",
                    {"page": "/signup", "source": "google", "campaign": "q4_growth"},
                ),
                (
                    "signup_form_started",
                    {"form_version": "v2", "ab_test": "new_layout"},
                ),
                ("email_entered", {"domain": "company.com", "validation_time_ms": 234}),
                ("password_created", {"strength": "strong", "requirements_met": 4}),
                (
                    "verification_email_sent",
                    {"email_provider": "gmail", "delivery_status": "pending"},
                ),
                (
                    "signup_completed",
                    {"time_to_complete_seconds": 127, "form_errors": 0},
                ),
                ("onboarding_tutorial_started", {"tutorial_version": "interactive_v3"}),
                (
                    "feature_flag_evaluated",
                    {"flag": "show_tutorial_tips", "value": True},
                ),
                ("tutorial_step_completed", {"step": 1, "time_spent_seconds": 45}),
                ("tutorial_step_completed", {"step": 2, "time_spent_seconds": 62}),
                ("tutorial_completed", {"completion_rate": 1.0, "feedback_score": 4.5}),
                (
                    "first_action_taken",
                    {"action_type": "create_project", "success": True},
                ),
            ]

            total_events = len(onboarding_events)
            for i, (event_name, properties) in enumerate(onboarding_events, 1):
                # Simulate realistic timing
                time.sleep(0.2)  # Small delay for demo purposes

                # Track event with governance
                if event_name == "feature_flag_evaluated":
                    # Special handling for feature flag evaluation
                    flag_value, metadata = (
                        adapter.evaluate_feature_flag_with_governance(
                            flag_key=properties["flag"],
                            distinct_id=f"user_{session.session_id[:8]}",
                            properties={
                                "signup_source": "organic",
                                "user_segment": "b2b",
                            },
                            session_id=session.session_id,
                        )
                    )
                    print(
                        f"  🚩 Evaluated feature flag '{properties['flag']}': {flag_value} - ${metadata['cost']:.6f}"
                    )
                else:
                    # Regular event tracking
                    result = adapter.capture_event_with_governance(
                        event_name=event_name,
                        properties=properties,
                        distinct_id=f"user_{session.session_id[:8]}",
                        is_identified=event_name
                        in ["signup_completed", "onboarding_tutorial_started"],
                        session_id=session.session_id,
                    )
                    print(f"  📊 Captured event '{event_name}': ${result['cost']:.6f}")

                # Progress indicator
                progress = i / total_events * 100
                print(
                    f"      Progress: [{int(progress / 5) * '█'}{(20 - int(progress / 5)) * '░'}] {progress:.1f}%"
                )

            print("\n📈 Analytics session completed successfully!")

    except Exception as e:
        print(f"❌ Analytics session failed: {e}")
        return

    # Display session summary
    print("\n💰 Session Cost Summary:")
    cost_summary = adapter.get_cost_summary()
    print(f"  Total Session Cost: ${cost_summary['daily_costs']:.4f}")
    print(f"  Events Tracked: {session.events_captured}")
    print(f"  Feature Flags Evaluated: {session.flags_evaluated}")
    print(f"  Cost per Event: ${session.total_cost / session.events_captured:.6f}")
    print(
        f"  Session Duration: {(session.end_time - session.start_time).total_seconds():.1f} seconds"
    )
    print(
        f"  Events per Second: {session.events_captured / (session.end_time - session.start_time).total_seconds():.2f}"
    )

    print("\n📊 Governance Metrics:")
    print(f"  Team: {cost_summary['team']}")
    print(f"  Project: {cost_summary['project']}")
    print(f"  Environment: {cost_summary['environment']}")
    print(
        f"  Daily Budget Utilization: {cost_summary['daily_budget_utilization']:.1f}%"
    )
    if session.customer_id:
        print(f"  Customer Attribution: {session.customer_id}")
    if session.cost_center:
        print(f"  Cost Center: {session.cost_center}")

    # Budget analysis
    daily_remaining = cost_summary["daily_budget_limit"] - cost_summary["daily_costs"]
    print("\n💳 Budget Analysis:")
    print(f"  Daily Budget: ${cost_summary['daily_budget_limit']:.2f}")
    print(f"  Used Today: ${cost_summary['daily_costs']:.4f}")
    print(f"  Remaining: ${daily_remaining:.4f}")

    if daily_remaining > 10:
        print("  💚 Budget Status: Healthy")
    elif daily_remaining > 1:
        print("  💛 Budget Status: Monitor usage")
    else:
        print("  🔴 Budget Status: Approaching limit")

    # Recommendations
    print("\n💡 Analytics Insights & Recommendations:")

    # Calculate some basic analytics
    avg_cost_per_event = (
        session.total_cost / session.events_captured
        if session.events_captured > 0
        else 0
    )
    events_per_dollar = 1 / avg_cost_per_event if avg_cost_per_event > 0 else 0

    print(f"  📈 Events per dollar: {events_per_dollar:,.0f}")
    print(
        f"  ⚡ Processing efficiency: {session.events_captured / (session.end_time - session.start_time).total_seconds():.1f} events/sec"
    )

    if events_per_dollar > 50000:
        print("  ✅ Excellent cost efficiency - you're in PostHog's free tier!")
    elif events_per_dollar > 10000:
        print("  👍 Good cost efficiency - optimized pricing tier")
    else:
        print("  💡 Consider volume discounts for higher event volumes")

    # Next steps
    print("\n🚀 Next Steps:")
    print("  1. Explore feature flags: python advanced_features.py")
    print("  2. Learn cost optimization: python cost_optimization.py")
    print("  3. See production patterns: python production_patterns.py")
    print("  4. Try auto-instrumentation: python auto_instrumentation.py")

    print("\n✅ Basic tracking example completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Basic tracking example interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print(
            "🐛 Please report this issue: https://github.com/KoshiHQ/GenOps-AI/issues"
        )
