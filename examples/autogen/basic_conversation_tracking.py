#!/usr/bin/env python3
"""
Basic AutoGen Conversation Tracking with GenOps Governance

This example demonstrates zero-code instrumentation of AutoGen conversations
with automatic cost tracking, conversation monitoring, and governance telemetry.

Features Demonstrated:
    - Zero-code auto-instrumentation setup
    - Conversation-level cost tracking
    - Agent interaction monitoring
    - Multi-provider cost aggregation
    - Real-time budget monitoring
    - Telemetry export to observability platforms

Usage:
    python examples/autogen/basic_conversation_tracking.py

Requirements:
    pip install pyautogen genops openai
    export OPENAI_API_KEY=your_key_here
"""

import logging
import os
from decimal import Decimal

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate basic AutoGen conversation tracking with GenOps."""

    print("🚀 AutoGen + GenOps Basic Conversation Tracking")
    print("=" * 60)

    # Step 1: Validate setup
    print("\n📋 Step 1: Validating setup...")
    try:
        from genops.providers.autogen import (
            print_validation_result,
            validate_autogen_setup,
        )

        result = validate_autogen_setup(
            team="demo-team", project="basic-conversation", verify_connectivity=True
        )
        print_validation_result(result, verbose=False)

        if not result.success:
            print("❌ Setup validation failed. Please fix issues before proceeding.")
            return

    except ImportError as e:
        print(f"❌ GenOps AutoGen integration not available: {e}")
        print("Install with: pip install genops")
        return

    # Step 2: Auto-instrument AutoGen
    print("\n🔧 Step 2: Setting up auto-instrumentation...")
    try:
        from genops.providers.autogen import auto_instrument

        adapter = auto_instrument(
            team="demo-team",
            project="basic-conversation",
            environment="development",
            daily_budget_limit=10.0,  # $10 daily limit for demo
            governance_policy="advisory",
        )

        print("✅ AutoGen auto-instrumentation enabled")
        print("   Team: demo-team")
        print("   Project: basic-conversation")
        print("   Daily Budget: $10.00")

    except Exception as e:
        print(f"❌ Failed to setup auto-instrumentation: {e}")
        return

    # Step 3: Create AutoGen agents (now automatically instrumented)
    print("\n🤖 Step 3: Creating AutoGen agents...")
    try:
        import autogen

        # Configure LLM (using OpenAI by default)
        config_list = [
            {"model": "gpt-3.5-turbo", "api_key": os.getenv("OPENAI_API_KEY")}
        ]

        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  No OpenAI API key found. Conversation will be simulated.")
            config_list = None

        # Create agents (automatically instrumented by GenOps)
        assistant = autogen.AssistantAgent(
            name="assistant",
            llm_config={"config_list": config_list} if config_list else False,
            system_message="You are a helpful AI assistant focused on providing clear, concise answers.",
        )

        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3,
            is_termination_msg=lambda x: (
                x.get("content", "").rstrip().endswith("TERMINATE")
            ),
            code_execution_config={
                "work_dir": "autogen_workspace",
                "use_docker": False,
            },
        )

        print("✅ Created instrumented AutoGen agents:")
        print(f"   - Assistant: {assistant.name}")
        print(f"   - User Proxy: {user_proxy.name}")

    except ImportError:
        print("❌ AutoGen not installed. Install with: pip install pyautogen")
        return
    except Exception as e:
        print(f"❌ Failed to create AutoGen agents: {e}")
        return

    # Step 4: Track conversation with GenOps
    print("\n💬 Step 4: Running tracked conversation...")
    try:
        with adapter.track_conversation(
            conversation_id="demo-chat", participants=["assistant", "user_proxy"]
        ) as context:
            # Start conversation (automatically tracked)
            user_proxy.initiate_chat(
                assistant,
                message="Hello! Can you explain what AutoGen is in simple terms? Keep it brief.",
            )

            # Simulate some metrics (in real usage, these would be automatic)
            context.add_turn(Decimal("0.002"), 150, "assistant")
            context.add_turn(Decimal("0.001"), 75, "user_proxy")

            print("✅ Conversation completed successfully")
            print(f"   Total cost: ${context.total_cost:.6f}")
            print(f"   Total turns: {context.turns_count}")

    except Exception as e:
        print(f"❌ Conversation tracking failed: {e}")
        logger.exception("Conversation error details:")

    # Step 5: Get session summary and insights
    print("\n📊 Step 5: Session summary and insights...")
    try:
        summary = adapter.get_session_summary()

        print("Session Summary:")
        print(f"   Total conversations: {summary['total_conversations']}")
        print(f"   Total cost: ${summary['total_cost']:.6f}")
        print(f"   Budget utilization: {summary['budget_utilization']:.1f}%")
        print(
            f"   Average cost per conversation: ${summary['avg_cost_per_conversation']:.6f}"
        )
        print(f"   Active agents: {', '.join(summary['active_agents'])}")

    except Exception as e:
        print(f"⚠️  Could not get session summary: {e}")

    # Step 6: Cost analysis and recommendations
    print("\n💰 Step 6: Cost analysis and optimization...")
    try:
        from genops.providers.autogen import analyze_conversation_costs

        analysis = analyze_conversation_costs(adapter, time_period_hours=1)

        if "error" not in analysis:
            print("Cost Analysis:")
            print(f"   Total cost: ${analysis['total_cost']:.6f}")
            print(f"   Cost by agent: {analysis['cost_by_agent']}")

            if analysis["recommendations"]:
                print("   Optimization recommendations:")
                for rec in analysis["recommendations"][:3]:  # Show top 3
                    print(f"   - {rec['reasoning']}")
        else:
            print(f"   Cost analysis: {analysis['error']}")

    except Exception as e:
        print(f"⚠️  Cost analysis not available: {e}")

    # Step 7: Cleanup
    print("\n🧹 Step 7: Cleanup...")
    try:
        from genops.providers.autogen import (
            disable_auto_instrumentation,
            get_instrumentation_stats,
        )

        # Show final stats
        stats = get_instrumentation_stats()
        print("Final instrumentation stats:")
        print(f"   Enabled: {stats['enabled']}")
        print(f"   Agents instrumented: {stats['stats'].get('agents_instrumented', 0)}")
        print(
            f"   Conversations tracked: {stats['stats'].get('conversations_tracked', 0)}"
        )

        # Disable instrumentation
        disable_auto_instrumentation()
        print("✅ Auto-instrumentation disabled")

    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")

    print("\n" + "=" * 60)
    print("🎉 Basic AutoGen conversation tracking completed!")
    print("\nKey achievements:")
    print("   ✅ Zero-code instrumentation setup")
    print("   ✅ Automatic conversation cost tracking")
    print("   ✅ Agent interaction monitoring")
    print("   ✅ Budget monitoring and alerts")
    print("   ✅ Cost optimization insights")
    print("\nNext steps:")
    print("   - Try multi_agent_group_chat_monitoring.py for group conversations")
    print("   - Explore production deployment patterns")
    print("   - Set up observability platform integration")
    print("=" * 60)


if __name__ == "__main__":
    main()
