#!/usr/bin/env python3
"""
Example 02: Auto-Instrumentation Patterns with Griptape

Complexity: ⭐ Beginner

This example demonstrates how GenOps auto-instrumentation works with existing
Griptape applications without requiring any code changes. Shows before/after
patterns and instrumentation management.

Prerequisites:
- Griptape framework installed (pip install griptape)
- GenOps installed (pip install genops)
- OpenAI API key set in environment

Usage:
    python 02_auto_instrumentation.py

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key
    GENOPS_TEAM: Team identifier for governance
    GENOPS_PROJECT: Project identifier (optional)
"""

import logging
import os

from griptape.rules import Rule
from griptape.structures import Agent, Pipeline
from griptape.tasks import PromptTask

# GenOps imports
from genops.providers.griptape import auto_instrument, disable_auto_instrument
from genops.providers.griptape.registration import (
    get_instrumentation_adapter,
    is_instrumented,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_existing_griptape_code():
    """
    Simulate existing Griptape application code.
    This code doesn't know about GenOps - it's your normal Griptape usage.
    """
    print("   Running existing Griptape Agent...")

    # Your existing Griptape code (unchanged)
    agent = Agent(
        tasks=[
            PromptTask(
                prompt="What are the key components of a modern AI application architecture?"
            )
        ],
        rules=[
            Rule("Provide a structured response with 3-4 main components"),
            Rule("Keep each point concise but informative"),
        ],
    )

    # Execute agent
    result = agent.run()
    print(f"   Agent response length: {len(result.output.value)} characters")

    return result


def run_existing_pipeline_code():
    """
    Simulate existing Griptape Pipeline code.
    Shows how pipelines are automatically instrumented too.
    """
    print("   Running existing Griptape Pipeline...")

    # Your existing Pipeline code (unchanged)
    pipeline = Pipeline(
        tasks=[
            PromptTask(
                id="analyze",
                prompt="Analyze the current state of AI governance: {{ input }}",
            ),
            PromptTask(
                id="recommendations",
                prompt="Based on this analysis: {{ analyze.output }}, provide 3 specific recommendations",
            ),
        ]
    )

    # Execute pipeline
    result = pipeline.run({"input": "Enterprise AI adoption is growing rapidly"})
    print(f"   Pipeline completed with {len(pipeline.tasks)} tasks")

    return result


def main():
    """Auto-instrumentation patterns demonstration."""

    print("🤖 GenOps + Griptape - Auto-Instrumentation Example")
    print("=" * 70)

    try:
        # Check environment
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            print("❌ Error: OPENAI_API_KEY environment variable is required")
            return False

        team = os.getenv("GENOPS_TEAM", "your-team")
        project = os.getenv("GENOPS_PROJECT", "griptape-demo")

        # === PART 1: Show code running without instrumentation ===
        print("\n📋 PART 1: Running WITHOUT GenOps Governance")
        print("-" * 50)

        print(
            "🔍 Instrumentation status:",
            "✅ Enabled" if is_instrumented() else "❌ Disabled",
        )

        print("📝 Executing existing Griptape code (no governance)...")
        run_existing_griptape_code()

        print("✅ Code executed normally - no governance tracking")

        # === PART 2: Enable auto-instrumentation ===
        print("\n📋 PART 2: Enabling GenOps Auto-Instrumentation")
        print("-" * 50)

        print("🚀 Enabling auto-instrumentation...")
        adapter = auto_instrument(
            team=team,
            project=project,
            environment="development",
            enable_cost_tracking=True,
            enable_performance_monitoring=True,
        )

        print(f"✅ Auto-instrumentation enabled for team '{team}'")
        print(
            "🔍 Instrumentation status:",
            "✅ Enabled" if is_instrumented() else "❌ Disabled",
        )

        # === PART 3: Show same code now with governance ===
        print("\n📋 PART 3: Running WITH GenOps Governance")
        print("-" * 50)

        print("📝 Executing SAME Griptape code (now with governance)...")
        run_existing_griptape_code()

        print("✅ Code executed with automatic governance tracking!")

        # Show governance data
        daily_spending = adapter.get_daily_spending()
        print(f"💰 Tracked spending: ${daily_spending:.6f}")

        # === PART 4: Show pipeline instrumentation ===
        print("\n📋 PART 4: Pipeline Auto-Instrumentation")
        print("-" * 50)

        print("📝 Executing Pipeline with auto-instrumentation...")
        run_existing_pipeline_code()

        # Updated spending
        new_daily_spending = adapter.get_daily_spending()
        pipeline_cost = new_daily_spending - daily_spending
        print(f"💰 Pipeline cost: ${pipeline_cost:.6f}")

        # === PART 5: Instrumentation management ===
        print("\n📋 PART 5: Instrumentation Management")
        print("-" * 50)

        print("🔧 Managing instrumentation state...")

        # Check current adapter
        current_adapter = get_instrumentation_adapter()
        if current_adapter:
            print(f"📊 Current adapter team: {current_adapter.governance_attrs.team}")
            print(
                f"📦 Current adapter project: {current_adapter.governance_attrs.project}"
            )

            # Get budget status
            budget_status = current_adapter.check_budget_compliance()
            print(f"💳 Budget status: {budget_status['status']}")

        # Demonstrate disabling (optional)
        print("\n🛑 Disabling auto-instrumentation...")
        disable_auto_instrument()
        print(
            "🔍 Instrumentation status:",
            "✅ Enabled" if is_instrumented() else "❌ Disabled",
        )

        # Re-enable for clean finish
        print("\n🔄 Re-enabling for demonstration...")
        auto_instrument(team=team, project=project, environment="development")
        print("🔍 Final status:", "✅ Enabled" if is_instrumented() else "❌ Disabled")

        # === SUMMARY ===
        print("\n🎉 Auto-Instrumentation Demo Complete!")
        print("\n✨ Key Takeaways:")
        print("  1. ✅ Zero code changes required for existing applications")
        print("  2. ✅ All Griptape structures automatically tracked")
        print("  3. ✅ Cost, performance, and governance added transparently")
        print("  4. ✅ Can be enabled/disabled dynamically")
        print("  5. ✅ Works with Agents, Pipelines, Workflows, and Engines")

        final_spending = (
            adapter.get_daily_spending() if is_instrumented() else new_daily_spending
        )
        print(f"\n💰 Total demo cost: ${final_spending:.6f}")

        print("\n🚀 Next Steps:")
        print("  • Try pipeline and workflow examples")
        print("  • Explore manual instrumentation for fine-grained control")
        print("  • Set up production deployment patterns")
        print("  • Configure your observability dashboard")

        return True

    except ImportError as e:
        if "griptape" in str(e):
            print("❌ Error: Griptape not installed")
            print("   Install with: pip install griptape")
        elif "genops" in str(e):
            print("❌ Error: GenOps not installed")
            print("   Install with: pip install genops")
        else:
            print(f"❌ Import error: {e}")
        return False

    except Exception as e:
        logger.error(f"Auto-instrumentation example failed: {e}")
        print(f"\n❌ Error occurred: {e}")
        print("\n🔧 Troubleshooting:")
        print("  • Verify your API keys are valid")
        print("  • Check network connectivity")
        print("  • Run validation script for detailed diagnostics")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
