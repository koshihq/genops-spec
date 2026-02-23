#!/usr/bin/env python3
"""
Raindrop AI + GenOps Zero-Code Auto-Instrumentation Example

This example demonstrates zero-code auto-instrumentation that automatically adds
GenOps governance, cost tracking, and telemetry to existing Raindrop AI workflows
without requiring any code changes.

Features demonstrated:
- Zero-code auto-instrumentation setup
- Automatic cost tracking for existing Raindrop workflows
- Transparent governance attribute injection
- Team and project attribution without code changes
- Budget monitoring and policy enforcement

Usage:
    export RAINDROP_API_KEY="your-api-key"
    export GENOPS_TEAM="your-team"
    export GENOPS_PROJECT="your-project"
    python auto_instrumentation.py

Author: GenOps AI Contributors
"""

import os
import sys
import time
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from genops.providers.raindrop import (  # noqa: F401
        auto_instrument,
        restore_raindrop,
    )
    from genops.providers.raindrop_validation import validate_setup
except ImportError as e:
    print(f"❌ Error importing GenOps Raindrop: {e}")
    print(
        "💡 Make sure you're in the project root directory and GenOps is properly installed"
    )
    sys.exit(1)


def main():
    """Demonstrate zero-code auto-instrumentation for Raindrop AI."""

    print("🚀 Raindrop AI + GenOps Zero-Code Auto-Instrumentation Example")
    print("=" * 70)

    # Get configuration
    api_key = os.getenv("RAINDROP_API_KEY")
    team = os.getenv("GENOPS_TEAM", "auto-instrumentation-team")
    project = os.getenv("GENOPS_PROJECT", "agent-monitoring-demo")
    environment = os.getenv("GENOPS_ENVIRONMENT", "development")

    print("\n📋 Configuration:")
    print(f"  Team: {team}")
    print(f"  Project: {project}")
    print(f"  Environment: {environment}")
    print(f"  API Key: {'✅ Configured' if api_key else '❌ Missing'}")

    # Quick validation
    validation_result = validate_setup(api_key)
    if not validation_result.is_valid:
        print("\n❌ Setup validation failed. Please check your configuration.")
        return

    print("\n🔄 Enabling auto-instrumentation for existing Raindrop workflows...")

    # Enable auto-instrumentation
    adapter = auto_instrument(
        raindrop_api_key=api_key,
        team=team,
        project=project,
        environment=environment,
        daily_budget_limit=25.0,  # Lower budget for demo
        enable_cost_alerts=True,
        governance_policy="advisory",
    )

    print("✅ Auto-instrumentation activated")
    print("\n📋 Your existing Raindrop code now includes:")
    print("  🏷️  Team and project attribution")
    print("  💰 Automatic cost tracking")
    print("  📊 Governance telemetry export")
    print("  🔍 Budget monitoring and alerts")
    print(f"  ⚖️  Policy enforcement ({adapter.governance_policy} mode)")

    # Example: Simulate existing Raindrop client usage
    print("\n🎯 Simulating existing Raindrop client usage...")
    print("(In a real scenario, this would be your existing Raindrop AI code)")

    try:
        # This would be your existing Raindrop AI code
        # Note: Since we don't have the actual Raindrop SDK installed,
        # we'll simulate the workflow and demonstrate the governance integration

        print("\n📝 Simulated Raindrop AI Operations:")
        print("-" * 40)

        # Simulate agent interactions (these would normally be Raindrop SDK calls)
        simulated_operations = [
            {
                "agent_id": "chatbot-v2",
                "interaction": "customer_query_1",
                "cost": 0.001,
                "performance": {"latency": 120, "accuracy": 0.94},
            },
            {
                "agent_id": "support-assistant",
                "interaction": "escalation_handling",
                "cost": 0.002,
                "performance": {"latency": 340, "accuracy": 0.91},
            },
            {
                "agent_id": "recommendation-engine",
                "interaction": "product_suggestion",
                "cost": 0.001,
                "performance": {"latency": 85, "accuracy": 0.96},
            },
        ]

        # Track each operation through the auto-instrumented adapter
        total_cost = 0

        with adapter.track_agent_monitoring_session(
            "auto_instrumented_session"
        ) as session:
            for i, operation in enumerate(simulated_operations, 1):
                # This simulates what would happen when your existing Raindrop code runs
                cost_result = session.track_agent_interaction(
                    agent_id=operation["agent_id"],
                    interaction_data={
                        "operation": operation["interaction"],
                        "performance_metrics": operation["performance"],
                        "auto_instrumented": True,
                    },
                    cost=operation["cost"],
                )

                print(
                    f"  ✅ Operation {i}: {operation['agent_id']} ({operation['interaction']}) - ${cost_result.total_cost:.3f}"
                )
                total_cost += float(cost_result.total_cost)
                time.sleep(0.1)  # Simulate processing delay

            # Show session summary
            print("\n📊 Auto-Instrumentation Summary:")
            print(f"  Operations Tracked: {session.operation_count}")
            print(f"  Total Cost: ${session.total_cost:.3f}")
            print(
                f"  Governance Attributes Added: {len(adapter.governance_attrs.to_dict())}"
            )
            print(f"  Telemetry Spans Created: {session.operation_count}")

            # Show governance attributes that were automatically added
            print("\n🏷️  Automatic Governance Attributes:")
            for key, value in adapter.governance_attrs.to_dict().items():
                print(f"    {key}: {value}")

    except Exception as e:
        print(f"⚠️  Simulated operation error (expected in demo): {e}")
        print("In a real scenario, this would be your actual Raindrop AI operations")

    # Demonstrate cost tracking benefits
    print("\n💡 Benefits of Auto-Instrumentation:")
    print("  ✅ Zero code changes required")
    print("  ✅ Automatic cost attribution to teams and projects")
    print("  ✅ Real-time budget monitoring")
    print("  ✅ OpenTelemetry-compatible telemetry export")
    print("  ✅ Policy enforcement without workflow disruption")
    print("  ✅ Enterprise governance compliance")

    # Show cost intelligence preview
    if adapter.cost_aggregator:
        summary = adapter.cost_aggregator.get_summary()
        if summary.total_cost > 0:
            print("\n📈 Cost Intelligence Preview:")
            print(f"  Total Sessions: {summary.session_count}")
            print(f"  Total Operations: {summary.total_operations}")
            print(f"  Total Cost: ${summary.total_cost:.3f}")
            print(
                f"  Average Cost per Operation: ${summary.average_cost_per_operation:.4f}"
            )

            # Show team/project breakdown
            if summary.cost_by_team:
                print("  Cost by Team:")
                for team_name, cost in summary.cost_by_team.items():
                    print(f"    {team_name}: ${cost:.3f}")

            if summary.cost_by_project:
                print("  Cost by Project:")
                for project_name, cost in summary.cost_by_project.items():
                    print(f"    {project_name}: ${cost:.3f}")

    # Integration patterns
    print("\n🔧 Integration Patterns:")
    print("  1. Web Applications: Add auto_instrument() to app startup")
    print("  2. Background Jobs: Enable at worker initialization")
    print("  3. Jupyter Notebooks: Run auto_instrument() in first cell")
    print("  4. CI/CD Pipelines: Include in deployment scripts")

    print("\n💡 Zero code changes required - existing workflows now governed!")

    # Cleanup (optional - demonstrates how to disable if needed)
    print("\n🧹 Cleanup (optional):")
    print("  To disable auto-instrumentation: restore_raindrop()")

    # Restore original behavior (optional)
    # restore_raindrop()
    # print("  ✅ Auto-instrumentation disabled")

    print("\n✅ Auto-instrumentation example completed successfully!")
    print("\n🚀 Next Steps:")
    print(
        "  1. Enable in your production code with: auto_instrument(team='your-team', project='your-project')"
    )
    print("  2. Configure your observability backend to receive OpenTelemetry data")
    print("  3. Set up dashboards and alerts for cost and governance monitoring")
    print("  4. Explore advanced features: python advanced_features.py")


if __name__ == "__main__":
    main()
