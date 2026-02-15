#!/usr/bin/env python3
"""
Example: Basic Flowise Flow Execution with Governance

Complexity: ⭐ Beginner

This example demonstrates the simplest way to execute a Flowise chatflow
with GenOps governance tracking. Perfect for getting started.

Prerequisites:
- Flowise instance running (local or cloud)
- At least one chatflow created in Flowise
- GenOps package installed

Usage:
    python 01_basic_flow_execution.py

Environment Variables:
    FLOWISE_BASE_URL: Flowise instance URL (default: http://localhost:3000)
    FLOWISE_API_KEY: API key (optional for local development)
    GENOPS_TEAM: Team name for cost attribution (default: flowise-examples)
"""

import logging
import os

from genops.providers.flowise import instrument_flowise
from genops.providers.flowise_validation import (
    print_validation_result,
    validate_flowise_setup,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate basic Flowise flow execution with governance."""

    print("🚀 Basic Flowise Flow Execution Example")
    print("=" * 50)

    # Configuration
    config = {
        "base_url": os.getenv("FLOWISE_BASE_URL", "http://localhost:3000"),
        "api_key": os.getenv("FLOWISE_API_KEY"),
        "team": os.getenv("GENOPS_TEAM", "flowise-examples"),
        "project": "basic-example",
        "environment": "development",
    }

    print(f"Flowise URL: {config['base_url']}")
    print(f"Team: {config['team']}")
    print(f"Project: {config['project']}")

    # Step 1: Validate setup
    print("\n📋 Step 1: Validating Flowise setup...")

    try:
        result = validate_flowise_setup(
            base_url=config["base_url"], api_key=config["api_key"]
        )

        if not result.is_valid:
            print("❌ Setup validation failed:")
            print_validation_result(result)
            return False

        print("✅ Setup validation passed!")

        if result.available_chatflows:
            print(f"Found {len(result.available_chatflows)} available chatflows:")
            for i, flow in enumerate(result.available_chatflows[:5], 1):
                print(f"  {i}. {flow}")

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False

    # Step 2: Create instrumented adapter
    print("\n⚙️ Step 2: Creating instrumented Flowise adapter...")

    try:
        flowise = instrument_flowise(**config)
        logger.info("Instrumented adapter created successfully")

    except Exception as e:
        logger.error(f"Failed to create adapter: {e}")
        return False

    # Step 3: Get available chatflows
    print("\n📊 Step 3: Fetching available chatflows...")

    try:
        chatflows = flowise.get_chatflows()

        if not chatflows:
            print(
                "❌ No chatflows found. Please create at least one chatflow in Flowise UI."
            )
            print("💡 Visit your Flowise instance and create a simple chatflow.")
            return False

        print(f"✅ Found {len(chatflows)} chatflows:")
        for flow in chatflows[:3]:  # Show first 3
            flow_id = flow.get("id", "unknown")
            flow_name = flow.get("name", "Unnamed")
            print(f"  • {flow_name} (ID: {flow_id})")

        # Use the first chatflow for our example
        selected_flow = chatflows[0]
        chatflow_id = selected_flow.get("id")
        chatflow_name = selected_flow.get("name", "Unnamed")

        print(f"\n🎯 Selected chatflow: {chatflow_name} (ID: {chatflow_id})")

    except Exception as e:
        logger.error(f"Failed to fetch chatflows: {e}")
        return False

    # Step 4: Execute chatflow with governance
    print(f"\n🤖 Step 4: Executing chatflow '{chatflow_name}'...")

    # Sample questions to test
    test_questions = [
        "Hello! How are you today?",
        "What can you help me with?",
        "Tell me about your capabilities.",
    ]

    successful_executions = 0

    for i, question in enumerate(test_questions, 1):
        print(f"\n  Question {i}: {question}")

        try:
            # Execute with governance tracking
            response = flowise.predict_flow(
                chatflow_id=chatflow_id,
                question=question,
                # Optional: Override governance attributes for this specific execution
                customer_id=f"example-customer-{i}",
                feature="basic-qa",
            )

            # Extract response text (format varies by chatflow type)
            response_text = ""
            if isinstance(response, dict):
                response_text = (
                    response.get("text")
                    or response.get("answer")
                    or response.get("content")
                    or str(response)
                )
            else:
                response_text = str(response)

            print(
                f"  ✅ Response: {response_text[:100]}{'...' if len(response_text) > 100 else ''}"
            )
            successful_executions += 1

        except Exception as e:
            logger.error(f"  ❌ Execution failed: {e}")
            continue

    # Step 5: Summary
    print("\n📈 Step 5: Execution Summary")
    print("=" * 30)
    print(f"Total questions: {len(test_questions)}")
    print(f"Successful executions: {successful_executions}")
    print(f"Success rate: {successful_executions / len(test_questions) * 100:.1f}%")

    if successful_executions > 0:
        print("\n✅ Governance tracking is working!")
        print("📊 Telemetry data has been captured for:")
        print("   • Cost attribution (team, project, customer)")
        print("   • Usage metrics (tokens, duration)")
        print("   • Performance tracking (execution time)")
        print("   • Error handling and debugging")

        print("\n💡 Next steps:")
        print("   • View telemetry in your observability platform")
        print("   • Try the auto-instrumentation example (02_auto_instrumentation.py)")
        print("   • Explore cost optimization (04_cost_optimization.py)")

    else:
        print("\n❌ All executions failed. Check:")
        print("   • Flowise is running and accessible")
        print("   • Chatflows are properly configured")
        print("   • API key is valid (if required)")

    return successful_executions > 0


def demo_governance_attributes():
    """Demonstrate different governance attribute patterns."""

    print("\n🏷️ Governance Attributes Demo")
    print("=" * 40)

    config = {
        "base_url": os.getenv("FLOWISE_BASE_URL", "http://localhost:3000"),
        "api_key": os.getenv("FLOWISE_API_KEY"),
    }

    # Different governance patterns
    governance_patterns = [
        {
            "name": "Team-based Attribution",
            "attrs": {
                "team": "customer-support",
                "project": "helpdesk-bot",
                "environment": "production",
            },
        },
        {
            "name": "Customer-based Attribution",
            "attrs": {
                "team": "saas-platform",
                "project": "customer-ai-assistant",
                "customer_id": "enterprise-customer-123",
                "cost_center": "product-engineering",
            },
        },
        {
            "name": "Feature-based Attribution",
            "attrs": {
                "team": "ai-research",
                "project": "nlp-experiments",
                "feature": "multilingual-support",
                "environment": "staging",
            },
        },
    ]

    for pattern in governance_patterns:
        print(f"\n📋 {pattern['name']}:")

        try:
            instrument_flowise(**config, **pattern["attrs"])

            # Show what attributes are being tracked
            for key, value in pattern["attrs"].items():
                print(f"   {key}: {value}")

            print("   ✅ Adapter created with governance attributes")

        except Exception as e:
            print(f"   ❌ Failed: {e}")


if __name__ == "__main__":
    try:
        print("Starting basic Flowise flow execution example...\n")

        # Run main example
        success = main()

        # Optional: Demonstrate governance patterns
        if success:
            demo_governance_attributes()

        print(
            f"\n{'✅ Example completed successfully!' if success else '❌ Example failed!'}"
        )

        if success:
            print("\n🎉 Congratulations! You've successfully:")
            print("   • Validated your Flowise setup")
            print("   • Created an instrumented Flowise adapter")
            print("   • Executed chatflows with governance tracking")
            print("   • Captured telemetry data for cost attribution")

            print("\n📚 Learn more:")
            print("   • Integration Guide: docs/integrations/flowise.md")
            print("   • More Examples: examples/flowise/")
            print("   • Auto-instrumentation: 02_auto_instrumentation.py")

        exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n⏹️ Example interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)
