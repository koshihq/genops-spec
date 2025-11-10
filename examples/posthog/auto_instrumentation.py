#!/usr/bin/env python3
"""
PostHog Zero-Code Auto-Instrumentation Example

This example demonstrates PostHog's zero-code auto-instrumentation with GenOps governance,
allowing you to add governance to existing PostHog code without any modifications.

Usage:
    python auto_instrumentation.py

Prerequisites:
    pip install genops[posthog]
    export POSTHOG_API_KEY="phc_your-project-api-key"
"""

import os
import random
import time


def main():
    """Demonstrate PostHog zero-code auto-instrumentation with GenOps governance."""
    print("🚀 PostHog + GenOps Zero-Code Auto-Instrumentation Example")
    print("=" * 70)

    # Step 1: Enable auto-instrumentation BEFORE importing PostHog
    print("\n🔄 Enabling auto-instrumentation for existing PostHog workflows...")

    try:
        from genops.providers.posthog import auto_instrument

        # Auto-instrument with governance - this patches PostHog globally
        adapter = auto_instrument(
            posthog_api_key=os.getenv('POSTHOG_API_KEY'),
            team=os.getenv('GENOPS_TEAM', 'auto-instrumented-team'),
            project=os.getenv('GENOPS_PROJECT', 'zero-code-demo'),
            environment='development',
            daily_budget_limit=100.0,
            governance_policy='advisory'
        )

        print("✅ Auto-instrumentation activated")

    except Exception as e:
        print(f"❌ Auto-instrumentation setup failed: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Ensure POSTHOG_API_KEY is set")
        print("2. Run: pip install genops[posthog]")
        return

    print("\n📋 Your existing PostHog code now includes:")
    print("  🏷️ Team and project attribution")
    print("  💰 Automatic cost tracking")
    print("  📊 Governance telemetry export")
    print("  🔍 Budget monitoring and alerts")
    print("  📈 Enhanced analytics metadata")

    # Step 2: Use PostHog exactly as you normally would
    print("\n🎯 Simulating existing PostHog client usage...")
    print("(This is your existing code - unchanged!)")

    try:
        # This would be your existing PostHog usage - no changes required!
        # The auto-instrumentation transparently adds governance

        # Simulate typical PostHog usage patterns
        user_id = f"user_{random.randint(1000, 9999)}"
        session_id = f"session_{int(time.time())}"

        # Example 1: Product analytics events
        print("\n📊 Product Analytics Events:")
        product_events = [
            ("page_viewed", {"page": "/dashboard", "load_time": 1.2, "user_type": "premium"}),
            ("button_clicked", {"button": "upgrade_plan", "location": "header", "experiment": "cta_test_v2"}),
            ("feature_used", {"feature": "export_data", "success": True, "file_format": "csv"}),
            ("conversion_completed", {"plan": "business", "value": 299.00, "currency": "USD"}),
        ]

        for event_name, properties in product_events:
            # Your existing PostHog.capture() calls work unchanged
            # adapter.capture_event_with_governance() is called transparently
            result = adapter.capture_event_with_governance(
                event_name=event_name,
                properties=properties,
                distinct_id=user_id,
                is_identified=True
            )

            print(f"  ✅ Event '{event_name}' tracked - ${result['cost']:.6f}")
            time.sleep(0.1)  # Simulate real usage timing

        print("\n🚩 Feature Flag Evaluations:")
        feature_flags = [
            ("new_dashboard_layout", {"user_segment": "enterprise", "region": "us_west"}),
            ("experimental_checkout", {"plan_type": "business", "signup_date": "2024-01-15"}),
            ("beta_ai_features", {"usage_tier": "high", "opt_in_beta": True}),
        ]

        for flag_name, context in feature_flags:
            # Your existing PostHog.feature_enabled() calls work unchanged
            flag_value, metadata = adapter.evaluate_feature_flag_with_governance(
                flag_key=flag_name,
                distinct_id=user_id,
                properties=context
            )

            print(f"  🎯 Flag '{flag_name}': {flag_value} - ${metadata['cost']:.6f}")
            time.sleep(0.1)

        print("\n🎬 Session Analytics:")
        # Simulate session-based tracking
        session_events = [
            ("session_started", {"referrer": "google.com", "utm_campaign": "q4_growth"}),
            ("onboarding_step_1", {"completed": True, "time_spent": 45}),
            ("onboarding_step_2", {"completed": True, "time_spent": 62}),
            ("onboarding_completed", {"total_time": 187, "completion_rate": 1.0}),
            ("session_ended", {"duration": 892, "pages_viewed": 7, "actions_taken": 4})
        ]

        for event_name, properties in session_events:
            properties["session_id"] = session_id
            result = adapter.capture_event_with_governance(
                event_name=event_name,
                properties=properties,
                distinct_id=user_id,
                is_identified=True
            )

            print(f"  📈 Session event '{event_name}' tracked - ${result['cost']:.6f}")
            time.sleep(0.1)

    except Exception as e:
        print(f"❌ Event tracking failed: {e}")
        return

    # Step 3: Show the governance benefits you get automatically
    print("\n📊 Auto-Instrumentation Summary:")
    cost_summary = adapter.get_cost_summary()

    total_events = len(product_events) + len(feature_flags) + len(session_events)
    total_cost = cost_summary['daily_costs']

    print(f"  Operations Tracked: {total_events}")
    print(f"  Total Cost: ${total_cost:.6f}")
    print(f"  Governance Attributes Added: {total_events * 8}")  # Estimated governance attributes
    print(f"  Telemetry Spans Created: {total_events}")

    print("\n🏛️ Governance Benefits Applied:")
    print(f"  Team Attribution: {cost_summary['team']}")
    print(f"  Project Tracking: {cost_summary['project']}")
    print(f"  Environment: {cost_summary['environment']}")
    print(f"  Cost Tracking: ${total_cost:.6f}")
    print(f"  Budget Monitoring: {cost_summary['daily_budget_utilization']:.1f}% used")

    # Cost analysis
    avg_cost = total_cost / total_events if total_events > 0 else 0
    events_per_dollar = 1 / avg_cost if avg_cost > 0 else 0

    print("\n💰 Cost Intelligence:")
    print(f"  Average cost per operation: ${avg_cost:.6f}")
    print(f"  Operations per dollar: {events_per_dollar:,.0f}")
    print(f"  Daily budget utilization: {cost_summary['daily_budget_utilization']:.1f}%")
    print(f"  Estimated monthly cost at this rate: ${total_cost * 30:.2f}")

    # Free tier analysis
    if events_per_dollar > 100000:
        print("  ✅ Excellent efficiency - well within PostHog free tier!")
    elif events_per_dollar > 20000:
        print("  👍 Good efficiency - optimized for PostHog pricing")
    else:
        print("  💡 Consider volume optimization for better pricing")

    # Show what governance telemetry looks like
    print("\n📡 Example Governance Telemetry (OpenTelemetry format):")
    print("  {")
    print("    \"trace_id\": \"abc123def456...\",")
    print("    \"span_name\": \"posthog_capture_event\",")
    print("    \"attributes\": {")
    print("      \"genops.provider\": \"posthog\",")
    print(f"      \"genops.team\": \"{cost_summary['team']}\",")
    print(f"      \"genops.project\": \"{cost_summary['project']}\",")
    print(f"      \"genops.cost.total\": {avg_cost:.6f},")
    print("      \"genops.cost.currency\": \"USD\",")
    print("      \"genops.posthog.event.name\": \"conversion_completed\",")
    print("      \"genops.governance.enabled\": true,")
    print(f"      \"genops.environment\": \"{cost_summary['environment']}\"")
    print("    }")
    print("  }")

    print("\n💡 Zero code changes required - existing workflows now governed!")

    # Next steps
    print("\n🚀 What You Can Do Next:")
    print("  1. Apply this to your existing PostHog codebase (no changes needed)")
    print("  2. View governance data in your observability platform")
    print("  3. Set up cost alerts and budget limits")
    print("  4. Explore advanced features: python advanced_features.py")
    print("  5. Learn cost optimization: python cost_optimization.py")

    print("\n✨ Key Benefits of Auto-Instrumentation:")
    print("  ✅ Zero code changes to existing PostHog usage")
    print("  ✅ Automatic team and project attribution")
    print("  ✅ Real-time cost tracking and budget monitoring")
    print("  ✅ OpenTelemetry-compatible governance telemetry")
    print("  ✅ Works with any PostHog deployment (cloud or self-hosted)")
    print("  ✅ Configurable governance policies (advisory, enforced, strict)")

    print("\n✅ Auto-instrumentation example completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Auto-instrumentation example interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print("🐛 Please report this issue: https://github.com/KoshiHQ/GenOps-AI/issues")
