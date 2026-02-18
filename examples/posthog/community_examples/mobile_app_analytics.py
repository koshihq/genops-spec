#!/usr/bin/env python3
"""
Mobile App Analytics with PostHog + GenOps

This example demonstrates mobile app analytics tracking with PostHog and GenOps
governance. It covers app lifecycle events, user engagement, feature usage,
performance monitoring, and in-app purchase tracking with cost intelligence.

Use Case:
    - iOS/Android mobile app user behavior tracking
    - App lifecycle and session management
    - Feature adoption and engagement analytics
    - Performance and crash reporting with governance
    - In-app purchase and subscription tracking

Usage:
    python community_examples/mobile_app_analytics.py

Prerequisites:
    pip install genops[posthog]
    export POSTHOG_API_KEY="phc_your_project_api_key"
    export GENOPS_TEAM="mobile-team"
    export GENOPS_PROJECT="mobile-app-analytics"

Expected Output:
    Complete mobile app session tracking with user engagement metrics,
    feature usage analytics, and performance monitoring with governance.

Learning Objectives:
    - Mobile app event taxonomy and lifecycle tracking
    - User engagement and retention analytics patterns
    - Performance monitoring with cost-aware telemetry
    - In-app purchase and subscription revenue tracking

Author: GenOps AI Community
License: Apache 2.0
"""

import random
import time


def main():
    """Demonstrate comprehensive mobile app analytics with PostHog + GenOps."""
    print("📱 Mobile App Analytics with PostHog + GenOps")
    print("=" * 50)
    print()

    # Import and setup GenOps PostHog adapter
    try:
        from genops.providers.posthog import GenOpsPostHogAdapter

        print("✅ GenOps PostHog integration loaded")
    except ImportError as e:
        print(f"❌ Failed to import GenOps PostHog: {e}")
        print("💡 Fix: pip install genops[posthog]")
        return False

    # Initialize mobile app analytics adapter
    print("\n🎯 Setting up Mobile App Analytics Configuration...")
    adapter = GenOpsPostHogAdapter(
        team="mobile-analytics",
        project="fitness-tracker-app",
        environment="production",
        customer_id="mobile_app_ios",
        cost_center="mobile_development",
        daily_budget_limit=75.0,  # Mobile apps typically have high event volumes
        governance_policy="advisory",  # Flexible for mobile event bursts
        tags={
            "app_platform": "ios",
            "app_version": "3.2.1",
            "analytics_tier": "standard",
            "crash_reporting": "enabled",
            "performance_monitoring": "enabled",
        },
    )

    print("✅ Mobile app adapter configured")
    print("   📱 Platform: iOS")
    print(f"   📊 Daily budget: ${adapter.daily_budget_limit}")
    print("   📈 App version: 3.2.1")
    print("   🔍 Performance monitoring: Enabled")

    # Mobile user segments for realistic simulation
    user_segments = [
        {
            "segment": "new_user",
            "session_length": (2, 8),  # minutes
            "feature_adoption": 0.3,
            "retention_day_1": 0.4,
        },
        {
            "segment": "active_user",
            "session_length": (5, 20),
            "feature_adoption": 0.7,
            "retention_day_1": 0.8,
        },
        {
            "segment": "power_user",
            "session_length": (15, 45),
            "feature_adoption": 0.9,
            "retention_day_1": 0.95,
        },
    ]

    # Simulate multiple mobile app sessions
    print("\n" + "=" * 50)
    print("📲 Simulating Mobile App User Sessions")
    print("=" * 50)

    total_sessions = 0
    total_events = 0
    total_revenue = 0.0
    feature_usage = {}

    for session_id in range(1, 6):  # 5 mobile app sessions
        segment = random.choice(user_segments)
        user_id = f"mobile_user_{session_id:03d}"
        device_info = generate_device_info()

        print(
            f"\n📱 Session #{session_id}: {segment['segment'].replace('_', ' ').title()}"
        )
        print("-" * 40)
        print(f"   Device: {device_info['model']} ({device_info['os_version']})")

        with adapter.track_analytics_session(
            session_name=f"mobile_session_{session_id}",
            customer_id=user_id,
            cost_center="mobile_user_acquisition",
            user_segment=segment["segment"],
            device_model=device_info["model"],
        ) as session:
            session_events = 0
            session_duration = random.randint(*segment["session_length"])

            # 1. App Launch and Initialization
            print("🚀 App Launch & Initialization")

            # App opened event
            result = adapter.capture_event_with_governance(
                event_name="app_opened",
                properties={
                    "app_version": "3.2.1",
                    "device_model": device_info["model"],
                    "os_version": device_info["os_version"],
                    "app_build": "3210",
                    "launch_time_ms": random.randint(800, 2500),
                    "cold_start": random.choice([True, False]),
                    "user_segment": segment["segment"],
                },
                distinct_id=user_id,
                session_id=session.session_id,
            )
            session_events += 1
            print(
                f"   ✅ App opened - Launch time: {result['properties'].get('launch_time_ms', 'N/A')}ms - Cost: ${result['cost']:.6f}"
            )

            # Screen views (core app navigation)
            screens = ["dashboard", "workout_list", "profile", "settings", "stats"]
            screens_visited = random.sample(screens, random.randint(2, len(screens)))

            for screen in screens_visited:
                result = adapter.capture_event_with_governance(
                    event_name="screen_viewed",
                    properties={
                        "screen_name": screen,
                        "previous_screen": screens_visited[
                            screens_visited.index(screen) - 1
                        ]
                        if screens_visited.index(screen) > 0
                        else "app_launch",
                        "view_duration_seconds": random.randint(5, 60),
                        "user_segment": segment["segment"],
                    },
                    distinct_id=user_id,
                    is_identified=True,  # Screen views are identified events
                    session_id=session.session_id,
                )
                session_events += 1
                print(f"   📺 Screen '{screen}' viewed - Cost: ${result['cost']:.6f}")

            # 2. Feature Usage and Engagement
            print("\n🎯 Feature Usage & Engagement")

            # Core feature usage based on user segment
            features = [
                {
                    "name": "workout_start",
                    "adoption_rate": 0.8,
                    "revenue_potential": 0.0,
                },
                {
                    "name": "progress_tracking",
                    "adoption_rate": 0.6,
                    "revenue_potential": 0.0,
                },
                {
                    "name": "social_sharing",
                    "adoption_rate": 0.3,
                    "revenue_potential": 0.0,
                },
                {
                    "name": "premium_workout",
                    "adoption_rate": 0.1,
                    "revenue_potential": 9.99,
                },
                {
                    "name": "nutrition_planner",
                    "adoption_rate": 0.2,
                    "revenue_potential": 4.99,
                },
            ]

            for feature in features:
                if (
                    random.random()
                    < feature["adoption_rate"] * segment["feature_adoption"]
                ):
                    feature_usage[feature["name"]] = (
                        feature_usage.get(feature["name"], 0) + 1
                    )

                    result = adapter.capture_event_with_governance(
                        event_name="feature_used",
                        properties={
                            "feature_name": feature["name"],
                            "usage_duration_seconds": random.randint(30, 300),
                            "user_segment": segment["segment"],
                            "feature_discovery": random.choice(
                                ["onboarding", "organic", "notification", "search"]
                            ),
                        },
                        distinct_id=user_id,
                        is_identified=True,
                        session_id=session.session_id,
                    )
                    session_events += 1
                    print(
                        f"   🔧 Feature '{feature['name']}' used - Cost: ${result['cost']:.6f}"
                    )

                    # In-app purchase simulation for premium features
                    if (
                        feature["revenue_potential"] > 0 and random.random() < 0.15
                    ):  # 15% purchase rate
                        result = adapter.capture_event_with_governance(
                            event_name="in_app_purchase",
                            properties={
                                "product_id": f"premium_{feature['name']}",
                                "price": feature["revenue_potential"],
                                "currency": "USD",
                                "purchase_type": "one_time",
                                "payment_method": "app_store",
                                "user_segment": segment["segment"],
                            },
                            distinct_id=user_id,
                            is_identified=True,
                            session_id=session.session_id,
                        )
                        session_events += 1
                        total_revenue += feature["revenue_potential"]
                        print(
                            f"   💰 In-app purchase: ${feature['revenue_potential']} - Cost: ${result['cost']:.6f}"
                        )

            # 3. Performance and Technical Events
            print("\n⚡ Performance & Technical Monitoring")

            # Performance metrics
            if random.random() < 0.7:  # 70% of sessions report performance
                result = adapter.capture_event_with_governance(
                    event_name="performance_metric",
                    properties={
                        "metric_type": "app_performance",
                        "cpu_usage_percent": random.uniform(10, 80),
                        "memory_usage_mb": random.randint(150, 400),
                        "battery_drain_percent": random.uniform(1, 5),
                        "network_requests": random.randint(5, 25),
                        "user_segment": segment["segment"],
                    },
                    distinct_id=user_id,
                    session_id=session.session_id,
                )
                session_events += 1
                print(
                    f"   📊 Performance metrics captured - Cost: ${result['cost']:.6f}"
                )

            # Error/crash reporting (low probability)
            if random.random() < 0.05:  # 5% chance of error
                error_types = [
                    "network_timeout",
                    "ui_freeze",
                    "data_sync_failed",
                    "crash",
                ]
                error_type = random.choice(error_types)

                result = adapter.capture_event_with_governance(
                    event_name="app_error",
                    properties={
                        "error_type": error_type,
                        "error_message": f"Mobile app {error_type} in session",
                        "stack_trace_available": random.choice([True, False]),
                        "user_segment": segment["segment"],
                        "app_state": random.choice(["foreground", "background"]),
                    },
                    distinct_id=user_id,
                    is_identified=True,
                    session_id=session.session_id,
                )
                session_events += 1
                print(
                    f"   ⚠️ App error '{error_type}' reported - Cost: ${result['cost']:.6f}"
                )

            # 4. Session End and Engagement
            print("\n👋 Session End & Engagement Summary")

            # Session completed
            result = adapter.capture_event_with_governance(
                event_name="session_ended",
                properties={
                    "session_duration_minutes": session_duration,
                    "screens_visited": len(screens_visited),
                    "features_used": len(
                        [f for f in features if f["name"] in feature_usage]
                    ),
                    "user_segment": segment["segment"],
                    "session_quality": "high" if session_duration > 10 else "standard",
                },
                distinct_id=user_id,
                session_id=session.session_id,
            )
            session_events += 1
            print(
                f"   ✅ Session ended - Duration: {session_duration}min - Cost: ${result['cost']:.6f}"
            )

            # App backgrounded
            result = adapter.capture_event_with_governance(
                event_name="app_backgrounded",
                properties={
                    "background_trigger": random.choice(
                        ["home_button", "notification", "phone_call", "app_switcher"]
                    ),
                    "session_duration_minutes": session_duration,
                    "user_segment": segment["segment"],
                },
                distinct_id=user_id,
                session_id=session.session_id,
            )
            session_events += 1
            print(f"   📱 App backgrounded - Cost: ${result['cost']:.6f}")

            total_sessions += 1
            total_events += session_events

            print("\n📊 Session Summary:")
            print(f"   Events in session: {session_events}")
            print(f"   Session duration: {session_duration} minutes")
            print(f"   Screens visited: {len(screens_visited)}")
            print(f"   User segment: {segment['segment'].replace('_', ' ').title()}")

            # Realistic mobile timing
            time.sleep(0.3)

    # Mobile app analytics summary
    print("\n" + "=" * 50)
    print("📈 Mobile App Analytics Summary")
    print("=" * 50)

    cost_summary = adapter.get_cost_summary()
    avg_session_length = sum(
        [random.randint(*seg["session_length"]) for seg in user_segments]
    ) / len(user_segments)

    print("📱 App Performance Metrics:")
    print(f"   Total sessions tracked: {total_sessions}")
    print(f"   Average session length: {avg_session_length:.1f} minutes")
    print(f"   Total events captured: {total_events}")
    print(f"   Events per session: {total_events / total_sessions:.1f}")
    print(f"   In-app revenue tracked: ${total_revenue:.2f}")

    print("\n🎯 Feature Adoption:")
    for feature, usage_count in feature_usage.items():
        adoption_rate = (usage_count / total_sessions) * 100
        print(
            f"   {feature.replace('_', ' ').title()}: {usage_count}/{total_sessions} sessions ({adoption_rate:.1f}%)"
        )

    print("\n💰 Cost Intelligence:")
    print(f"   Total analytics cost: ${cost_summary['daily_costs']:.6f}")
    print(f"   Cost per session: ${cost_summary['daily_costs'] / total_sessions:.6f}")
    print(f"   Cost per event: ${cost_summary['daily_costs'] / total_events:.6f}")
    print(f"   Budget utilization: {cost_summary['daily_budget_utilization']:.1f}%")

    print("\n🏛️ Mobile Governance:")
    print(f"   Team: {cost_summary['team']}")
    print(f"   Project: {cost_summary['project']}")
    print(f"   Environment: {cost_summary['environment']}")
    print("   Platform tracking: iOS/Android")
    print("   Performance monitoring: Enabled")

    # Mobile-specific insights
    print("\n📊 Mobile App Insights:")
    if total_revenue > 0:
        print(
            f"   Revenue per analytics dollar: ${total_revenue / cost_summary['daily_costs']:.2f}"
        )
        print(f"   Analytics ROI: {(total_revenue / cost_summary['daily_costs']):.0f}x")
    print(
        f"   Estimated monthly app analytics cost: ${cost_summary['daily_costs'] * 30:.2f}"
    )
    print(
        f"   Cost efficiency: ${cost_summary['daily_costs'] / total_events * 1000:.3f} per 1K events"
    )

    print("\n✅ Mobile app analytics tracking completed successfully!")
    return True


def generate_device_info() -> dict[str, str]:
    """Generate realistic mobile device information."""
    ios_devices = [
        {"model": "iPhone 14 Pro", "os_version": "iOS 16.4"},
        {"model": "iPhone 13", "os_version": "iOS 16.3"},
        {"model": "iPhone 12", "os_version": "iOS 16.2"},
        {"model": "iPad Air", "os_version": "iOS 16.4"},
    ]

    android_devices = [
        {"model": "Samsung Galaxy S23", "os_version": "Android 13"},
        {"model": "Google Pixel 7", "os_version": "Android 13"},
        {"model": "OnePlus 11", "os_version": "Android 13"},
        {"model": "Samsung Galaxy Tab", "os_version": "Android 12"},
    ]

    all_devices = ios_devices + android_devices
    return random.choice(all_devices)


def get_mobile_analytics_recommendations() -> list[dict[str, str]]:
    """Generate mobile app analytics optimization recommendations."""
    return [
        {
            "category": "User Retention",
            "recommendation": "Track user lifecycle stages for personalized onboarding",
            "implementation": "Add user_lifecycle_stage to all events (new, activated, retained, churned)",
            "expected_impact": "25-40% improvement in Day 1 retention",
        },
        {
            "category": "Performance Optimization",
            "recommendation": "Implement smart event batching for battery efficiency",
            "implementation": "Batch non-critical events and send during charging/WiFi",
            "expected_impact": "60-80% reduction in battery impact",
        },
        {
            "category": "Cost Optimization",
            "recommendation": "Use local analytics SDK with intelligent sync",
            "implementation": "Cache events locally and sync based on connectivity/cost",
            "expected_impact": "40-60% reduction in analytics costs",
        },
        {
            "category": "Feature Discovery",
            "recommendation": "Track feature discovery paths for UX optimization",
            "implementation": "Add discovery_method to all feature_used events",
            "expected_impact": "20-35% increase in feature adoption",
        },
    ]


if __name__ == "__main__":
    try:
        success = main()

        if success:
            print("\n💡 Mobile Analytics Best Practices:")
            recommendations = get_mobile_analytics_recommendations()
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec['category']}: {rec['recommendation']}")
                print(f"      Implementation: {rec['implementation']}")
                print(f"      Expected Impact: {rec['expected_impact']}")
                print()

        exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n👋 Mobile analytics demonstration interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Error in mobile analytics example: {e}")
        print("🔧 Please check your PostHog configuration and try again")
        exit(1)
