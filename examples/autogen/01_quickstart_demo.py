#!/usr/bin/env python3
"""
AutoGen + GenOps: 3-Minute Quickstart Demo

This is the fastest way to see AutoGen + GenOps governance in action.
Demonstrates the exact 3-step process from the quickstart guide.

Features Demonstrated:
    - One-line governance setup
    - Zero code changes to existing AutoGen
    - Immediate cost tracking and insights
    - Built-in validation and troubleshooting

Usage:
    python examples/autogen/01_quickstart_demo.py

Prerequisites:
    pip install genops[autogen]
    export OPENAI_API_KEY=your_key  # or any LLM provider
"""

import os


def demo_3_step_quickstart():
    """Demonstrate the exact 3-step quickstart process."""

    print("🚀 AutoGen + GenOps: 3-Step Quickstart Demo")
    print("=" * 50)

    # Step 1: Installation check (simulated - user runs pip install genops[autogen])
    print("\n📦 Step 1: Installation")
    print("✅ genops[autogen] - Assumed installed")
    print("   (Run: pip install genops[autogen])")

    # Step 2: Quick validation
    print("\n🔍 Step 2: Quick Validation (30 seconds)")
    try:
        from genops.providers.autogen import quick_validate

        result = quick_validate()
        print(
            "✅ Environment validated!" if result else "⚠️  Issues found - check setup"
        )
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        print("💡 Fix: Ensure GenOps is installed: pip install genops[autogen]")
        return

    # Step 3: One-line governance setup
    print("\n⚙️  Step 3: Enable Governance (1 line of code)")

    try:
        # The magic one-liner from the quickstart!
        from genops.providers.autogen import enable_governance

        enable_governance()
        print("✅ Governance enabled with one line!")

    except Exception as e:
        print(f"❌ Governance setup failed: {e}")
        return

    # Now demonstrate that existing AutoGen code works unchanged
    print("\n🤖 Demo: Your Existing AutoGen Code (Unchanged)")
    print("Creating AutoGen agents...")

    try:
        import autogen

        # Your existing AutoGen setup (completely unchanged!)
        config_list = [
            {
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY", "demo-key"),
            }
        ]

        # Skip actual LLM calls if no API key for demo purposes
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  No OPENAI_API_KEY found - simulating conversation")
            config_list = False

        assistant = autogen.AssistantAgent(
            name="assistant",
            llm_config={"config_list": config_list} if config_list else False,
            system_message="You are a helpful assistant.",
        )

        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
            is_termination_msg=lambda x: (
                x.get("content", "").rstrip().endswith("TERMINATE")
            ),
        )

        print("✅ AutoGen agents created successfully")
        print("   - Assistant agent with governance tracking")
        print("   - User proxy with automatic cost attribution")

        # Simulate conversation (or run real one if API key available)
        print("\n💬 Running Conversation...")
        if config_list:
            print("   Starting actual AutoGen conversation...")
            user_proxy.initiate_chat(
                assistant,
                message="Hello! Can you briefly explain what AutoGen is? Keep it under 50 words.",
            )
        else:
            print(
                "   [Simulated] Assistant: AutoGen is a Microsoft framework for multi-agent"
            )
            print(
                "   [Simulated] conversations where AI agents collaborate to solve tasks."
            )
            print("   [Simulated] Conversation completed!")

        print("✅ Conversation completed with automatic governance tracking!")

    except ImportError:
        print("❌ AutoGen not available")
        print("💡 Fix: pip install pyautogen")
        return
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return

    # Show what governance data was captured
    print("\n📊 Governance Data Captured")
    try:
        from genops.providers.autogen import get_current_adapter

        adapter = get_current_adapter()
        if adapter:
            summary = adapter.get_session_summary()
            print("✅ Session tracked successfully:")
            print(f"   - Total conversations: {summary.get('total_conversations', 0)}")
            print(f"   - Total cost: ${summary.get('total_cost', 0):.6f}")
            print(
                f"   - Budget utilization: {summary.get('budget_utilization', 0):.1f}%"
            )
            print(f"   - Active agents: {len(summary.get('active_agents', []))}")
        else:
            print("⚠️  Adapter not available (expected in simulation mode)")

    except Exception as e:
        print(f"⚠️  Could not retrieve governance data: {e}")

    # Success message
    print("\n" + "=" * 50)
    print("🎉 SUCCESS: 3-Step Quickstart Complete!")
    print("\nWhat just happened:")
    print("✅ Added comprehensive AutoGen governance in 3 steps")
    print("✅ Zero changes to your existing AutoGen code")
    print("✅ Automatic cost tracking and budget monitoring")
    print("✅ Enterprise-grade telemetry and observability")

    print("\n🚀 Next Steps:")
    print("1. Set your API key: export OPENAI_API_KEY=your_key")
    print("2. Try more examples: python examples/autogen/02_conversation_tracking.py")
    print("3. Read comprehensive guide: docs/integrations/autogen.md")
    print("4. Join the community: https://github.com/KoshiHQ/GenOps-AI")
    print("=" * 50)


def show_code_example():
    """Show the exact code pattern users can copy."""
    print("\n📋 Copy-Paste Code Template:")
    print("-" * 30)
    print("""
# Add this ONE line to any existing AutoGen script:
from genops.providers.autogen import enable_governance; enable_governance()

# Your existing AutoGen code works unchanged:
import autogen

config_list = [{"model": "gpt-3.5-turbo", "api_key": "your-key"}]
assistant = autogen.AssistantAgent(name="assistant", llm_config={"config_list": config_list})
user_proxy = autogen.UserProxyAgent(name="user", human_input_mode="NEVER")

user_proxy.initiate_chat(assistant, message="Hello!")
# ↑ Now tracked with comprehensive governance!
""")
    print("-" * 30)


if __name__ == "__main__":
    demo_3_step_quickstart()
    show_code_example()
