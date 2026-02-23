#!/usr/bin/env python3
"""
Example 01: Basic Griptape Agent with GenOps Governance

Complexity: ⭐ Beginner

This example demonstrates the simplest way to add GenOps governance to a Griptape Agent.
Shows automatic cost tracking, team attribution, and telemetry generation.

Prerequisites:
- Griptape framework installed (pip install griptape)
- GenOps installed (pip install genops)
- OpenAI API key set in environment
- GENOPS_TEAM environment variable set

Usage:
    python 01_basic_agent.py

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key
    GENOPS_TEAM: Team identifier for governance
    GENOPS_PROJECT: Project identifier (optional)
"""

import logging
import os

from griptape.rules import Rule
from griptape.structures import Agent
from griptape.tasks import PromptTask

# Import GenOps auto-instrumentation
from genops.providers.griptape import auto_instrument

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Basic Agent example with GenOps governance."""

    print("🤖 GenOps + Griptape - Basic Agent Example")
    print("=" * 60)

    try:
        # Check environment variables
        openai_key = os.getenv("OPENAI_API_KEY")
        team = os.getenv("GENOPS_TEAM", "your-team")
        project = os.getenv("GENOPS_PROJECT", "griptape-demo")

        if not openai_key:
            print("❌ Error: OPENAI_API_KEY environment variable is required")
            print("   Set it with: export OPENAI_API_KEY='your-api-key'")
            return False

        # Enable GenOps governance (1 line!)
        print("📊 Enabling GenOps governance...")
        adapter = auto_instrument(team=team, project=project, environment="development")
        print(f"✅ Governance enabled for team '{team}', project '{project}'")

        # Create a basic Griptape Agent
        print("\n🚀 Creating Griptape Agent...")
        agent = Agent(
            tasks=[
                PromptTask(
                    prompt="Explain the benefits of AI governance in 2-3 clear sentences. Focus on practical value for development teams."
                )
            ],
            rules=[
                Rule("Keep response concise and professional"),
                Rule("Focus on practical benefits, not theory"),
                Rule("Use specific examples where possible"),
            ],
        )
        print("✅ Agent created successfully")

        # Execute agent with automatic governance tracking
        print("\n🎯 Executing Agent with GenOps tracking...")
        result = agent.run()

        print("\n📝 Agent Response:")
        print("-" * 40)
        print(result.output.value)
        print("-" * 40)

        # Show governance metrics
        print("\n📊 GenOps Governance Metrics:")
        daily_spending = adapter.get_daily_spending()
        budget_status = adapter.check_budget_compliance()

        print(f"  💰 Daily Spending: ${daily_spending:.6f}")
        print(f"  📈 Budget Status: {budget_status['status']}")
        print(f"  👥 Team: {adapter.governance_attrs.team}")
        print(f"  📦 Project: {adapter.governance_attrs.project}")
        print(f"  🌍 Environment: {adapter.governance_attrs.environment}")

        if budget_status.get("utilization"):
            print(f"  📊 Budget Utilization: {budget_status['utilization']:.1f}%")

        print("\n🎉 Example completed successfully!")
        print("\n✨ What just happened:")
        print("  1. ✅ GenOps auto-instrumentation enabled")
        print("  2. ✅ Griptape Agent executed with governance")
        print("  3. ✅ Cost and usage automatically tracked")
        print("  4. ✅ Team and project attribution added")
        print("  5. ✅ OpenTelemetry telemetry generated")

        print("\n🚀 Next Steps:")
        print("  • Try example 02: Auto-instrumentation patterns")
        print("  • Explore the complete integration guide")
        print("  • Set up your observability dashboard")

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
        logger.error(f"Example failed: {e}")
        print(f"\n❌ Error occurred: {e}")
        print("\n🔧 Troubleshooting Tips:")
        print("  • Check your OpenAI API key is valid")
        print("  • Ensure you have internet connectivity")
        print("  • Run the setup validation script")
        print("  • Check the troubleshooting guide in the documentation")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
