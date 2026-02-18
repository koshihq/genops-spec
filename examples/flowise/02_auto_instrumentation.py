#!/usr/bin/env python3
"""
Example: Zero-Code Auto-Instrumentation

Complexity: ⭐ Beginner

This example demonstrates GenOps auto-instrumentation for Flowise, which
automatically tracks all Flowise API calls with zero code changes to your
existing application.

Prerequisites:
- Flowise instance running
- Existing Flowise application code (or requests-based code)
- GenOps package installed

Usage:
    python 02_auto_instrumentation.py

Environment Variables:
    FLOWISE_BASE_URL: Flowise instance URL
    FLOWISE_API_KEY: API key (optional for local dev)
    GENOPS_TEAM: Team name for governance
"""

import logging
import os
import time

import requests

from genops.providers.flowise import auto_instrument, disable_auto_instrument
from genops.providers.flowise_validation import validate_flowise_setup

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def simulate_existing_flowise_application(base_url: str, chatflow_id: str):
    """
    Simulate existing Flowise application code that would benefit from
    auto-instrumentation without requiring any code changes.
    """

    print("\n🔄 Simulating existing Flowise application...")

    # This represents your existing Flowise application code
    # No GenOps-specific code - just standard HTTP requests

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    # Simulate various types of Flowise API calls
    api_calls = [
        {
            "name": "List Chatflows",
            "method": "GET",
            "url": f"{base_url}/api/v1/chatflows",
        },
        {
            "name": "Get Specific Chatflow",
            "method": "GET",
            "url": f"{base_url}/api/v1/chatflows/{chatflow_id}",
        },
        {
            "name": "Predict Flow - Customer Inquiry",
            "method": "POST",
            "url": f"{base_url}/api/v1/prediction/{chatflow_id}",
            "json": {
                "question": "What are your business hours?",
                "sessionId": "customer-session-001",
            },
        },
        {
            "name": "Predict Flow - Technical Support",
            "method": "POST",
            "url": f"{base_url}/api/v1/prediction/{chatflow_id}",
            "json": {
                "question": "How do I reset my password?",
                "sessionId": "customer-session-002",
            },
        },
        {
            "name": "Predict Flow - Product Information",
            "method": "POST",
            "url": f"{base_url}/api/v1/prediction/{chatflow_id}",
            "json": {
                "question": "Tell me about your premium features.",
                "sessionId": "customer-session-003",
            },
        },
    ]

    results = []

    for call in api_calls:
        print(f"  📡 Making API call: {call['name']}")

        try:
            if call["method"] == "GET":
                response = session.get(call["url"])
            elif call["method"] == "POST":
                response = session.post(call["url"], json=call.get("json"))

            response.raise_for_status()

            print(f"     ✅ Success: {response.status_code}")
            results.append(
                {"call": call["name"], "status": response.status_code, "success": True}
            )

            # Brief delay to simulate real application behavior
            time.sleep(0.5)

        except Exception as e:
            print(f"     ❌ Failed: {e}")
            results.append({"call": call["name"], "error": str(e), "success": False})

    return results


def demonstrate_auto_instrumentation():
    """Demonstrate auto-instrumentation setup and benefits."""

    print("🔧 Auto-Instrumentation Demonstration")
    print("=" * 50)

    # Configuration
    base_url = os.getenv("FLOWISE_BASE_URL", "http://localhost:3000")
    api_key = os.getenv("FLOWISE_API_KEY")
    team = os.getenv("GENOPS_TEAM", "auto-instrumentation-demo")
    project = "zero-code-example"

    print(f"Flowise URL: {base_url}")
    print(f"Team: {team}")
    print(f"Project: {project}")

    # Step 1: Validate setup
    print("\n📋 Step 1: Validating Flowise setup...")

    try:
        result = validate_flowise_setup(base_url, api_key)

        if not result.is_valid:
            print("❌ Setup validation failed. Please fix issues before continuing.")
            return False

        print("✅ Setup validation passed!")

        if not result.available_chatflows:
            print("❌ No chatflows available for testing.")
            return False

        chatflow_id = None
        # Try to get a chatflow ID from available flows
        if result.available_chatflows:
            # For demo purposes, we'll need to get the actual chatflow ID
            # In a real scenario, you'd have this from your application
            from genops.providers.flowise import instrument_flowise

            temp_flowise = instrument_flowise(base_url=base_url, api_key=api_key)
            chatflows = temp_flowise.get_chatflows()
            if chatflows:
                chatflow_id = chatflows[0].get("id")
                chatflow_name = chatflows[0].get("name", "Unnamed")
                print(f"Using chatflow: {chatflow_name} (ID: {chatflow_id})")

        if not chatflow_id:
            print("❌ Cannot determine chatflow ID for demo.")
            return False

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False

    # Step 2: Show "before" - application without instrumentation
    print("\n📊 Step 2: Running application WITHOUT instrumentation...")
    print("(This represents your existing code)")

    before_results = simulate_existing_flowise_application(base_url, chatflow_id)

    successful_before = sum(1 for r in before_results if r["success"])
    print(f"Results: {successful_before}/{len(before_results)} calls successful")
    print("❗ No governance tracking - costs and usage not captured!")

    # Step 3: Enable auto-instrumentation
    print("\n⚡ Step 3: Enabling auto-instrumentation...")
    print("🎯 This is the ONLY code change needed!")

    print("\n--- CODE CHANGE ---")
    print("from genops.providers.flowise import auto_instrument")
    print("")
    print("# Add this single line at application startup:")
    print("auto_instrument(")
    print(f"    team='{team}',")
    print(f"    project='{project}',")
    print("    environment='development',")
    print("    enable_console_export=True  # Show telemetry in console")
    print(")")
    print("--- END CODE CHANGE ---\n")

    try:
        success = auto_instrument(
            base_url=base_url,
            api_key=api_key,
            team=team,
            project=project,
            environment="development",
            enable_console_export=True,  # Show telemetry in console for demo
            customer_id="demo-customer",
            cost_center="engineering",
        )

        if success:
            print("✅ Auto-instrumentation enabled successfully!")
            print("   All HTTP requests to Flowise will now be tracked automatically.")
        else:
            print("❌ Auto-instrumentation failed to initialize.")
            return False

    except Exception as e:
        logger.error(f"Auto-instrumentation failed: {e}")
        return False

    # Step 4: Show "after" - same application code, now with instrumentation
    print("\n📈 Step 4: Running SAME application WITH instrumentation...")
    print("(Exact same code as before - zero changes to your application!)")

    after_results = simulate_existing_flowise_application(base_url, chatflow_id)

    successful_after = sum(1 for r in after_results if r["success"])
    print(f"Results: {successful_after}/{len(after_results)} calls successful")
    print("✅ Full governance tracking now active!")

    # Step 5: Demonstrate what's being tracked
    print("\n📊 Step 5: What's being tracked automatically:")
    print("=" * 45)

    tracked_metrics = [
        "🏷️  Team Attribution: All costs attributed to your team",
        "💰 Cost Tracking: Automatic cost calculation per request",
        "⏱️  Performance: Request duration and response times",
        "🔍 Usage Metrics: Token estimates and API usage patterns",
        "🏢 Multi-Tenant: Customer-specific cost allocation",
        "📈 Observability: OpenTelemetry export to your platform",
        "🚨 Error Tracking: Failed requests and error rates",
        "🔄 Session Tracking: Conversation continuity monitoring",
    ]

    for metric in tracked_metrics:
        print(f"   {metric}")

    print("\n🎯 Benefits of Auto-Instrumentation:")
    print("   • Zero code changes to existing application")
    print("   • Automatic governance for all Flowise API calls")
    print("   • Works with any HTTP client (requests, httpx, urllib)")
    print("   • Compatible with existing observability tools")
    print("   • Easy to enable/disable without code changes")

    # Step 6: Show how to disable (optional)
    print("\n🔧 Step 6: Managing auto-instrumentation...")

    print("\nTo disable auto-instrumentation (if needed):")
    print("```python")
    print("from genops.providers.flowise import disable_auto_instrument")
    print("disable_auto_instrument()")
    print("```")

    return successful_after > 0


def advanced_auto_instrumentation_patterns():
    """Show advanced patterns for auto-instrumentation."""

    print("\n🔬 Advanced Auto-Instrumentation Patterns")
    print("=" * 50)

    patterns = [
        {
            "name": "Environment-Specific Configuration",
            "code": """
# Different configs per environment
if os.getenv('ENVIRONMENT') == 'production':
    auto_instrument(
        team="production-team",
        project="customer-service",
        environment="production",
        cost_center="operations"
    )
elif os.getenv('ENVIRONMENT') == 'staging':
    auto_instrument(
        team="staging-team",
        project="customer-service",
        environment="staging",
        enable_console_export=True
    )
else:  # development
    auto_instrument(
        team="dev-team",
        project="customer-service",
        environment="development",
        enable_console_export=True
    )
""",
        },
        {
            "name": "Multi-Application Setup",
            "code": """
# Different applications using same Flowise instance
# App 1: Customer Support
auto_instrument(
    team="customer-support",
    project="helpdesk-automation",
    feature="automated-responses"
)

# App 2: Sales Assistant
auto_instrument(
    team="sales",
    project="lead-qualification",
    feature="sales-ai-assistant"
)
""",
        },
        {
            "name": "Dynamic Attribute Assignment",
            "code": """
# Use request context for dynamic attributes
import threading

# Store per-request context
request_context = threading.local()

def set_request_context(customer_id, user_tier):
    request_context.customer_id = customer_id
    request_context.user_tier = user_tier

# Auto-instrumentation will pick up dynamic attributes
auto_instrument(
    team="saas-platform",
    project="multi-tenant-ai",
    # These will be set dynamically per request
    attribute_provider=lambda: {
        'customer_id': getattr(request_context, 'customer_id', None),
        'user_tier': getattr(request_context, 'user_tier', 'free')
    }
)
""",
        },
    ]

    for pattern in patterns:
        print(f"\n📋 {pattern['name']}:")
        print(pattern["code"])

    print("\n💡 Best Practices:")
    print("   • Enable auto-instrumentation once at application startup")
    print("   • Use environment variables for configuration")
    print("   • Set meaningful team/project names for cost attribution")
    print("   • Enable console export for development/debugging")
    print("   • Use disable_auto_instrument() for testing scenarios")


def main():
    """Main example function."""

    try:
        # Run the main demonstration
        success = demonstrate_auto_instrumentation()

        if success:
            # Show advanced patterns
            advanced_auto_instrumentation_patterns()

            print("\n🎉 Auto-Instrumentation Example Complete!")
            print("=" * 50)
            print("✅ You've learned how to:")
            print("   • Enable zero-code auto-instrumentation")
            print("   • Track existing Flowise applications automatically")
            print("   • Capture comprehensive governance telemetry")
            print("   • Set up team and project attribution")
            print("   • Export data to observability platforms")

            print("\n📚 Next Steps:")
            print("   • Try multi-flow orchestration (03_multi_flow_orchestration.py)")
            print("   • Explore cost optimization (04_cost_optimization.py)")
            print("   • Set up production monitoring (07_production_monitoring.py)")

            # Clean up: disable auto-instrumentation
            print("\n🧹 Cleaning up: Disabling auto-instrumentation...")
            disable_auto_instrument()
            print("✅ Auto-instrumentation disabled")

        return success

    except Exception as e:
        logger.error(f"Example failed: {e}")
        return False


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Example interrupted by user")
        # Clean up
        disable_auto_instrument()
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)
