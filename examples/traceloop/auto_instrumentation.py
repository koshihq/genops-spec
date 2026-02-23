#!/usr/bin/env python3
"""
Auto-Instrumentation OpenLLMetry + GenOps Example

This example demonstrates zero-code governance enhancement for existing OpenLLMetry applications.
GenOps automatically adds cost attribution, team tracking, and policy enforcement without
requiring changes to your existing code.

Perfect for teams already using OpenLLMetry who want to add governance intelligence.

Usage:
    python auto_instrumentation.py

Prerequisites:
    pip install genops[traceloop]  # Includes OpenLLMetry
    export OPENAI_API_KEY="your-openai-api-key"

    # Optional: For Traceloop commercial platform
    export TRACELOOP_API_KEY="your-traceloop-api-key"
"""

import asyncio
import os
from datetime import datetime


def setup_auto_instrumentation():
    """
    Set up automatic instrumentation that enhances existing OpenLLMetry code
    with GenOps governance without requiring code changes.
    """
    print("⚡ Auto-Instrumentation Setup")
    print("=" * 30)

    try:
        # Import and initialize GenOps auto-instrumentation
        from genops.providers.traceloop import auto_instrument

        print("✅ GenOps auto-instrumentation loaded")

        # Configure governance context for all operations
        governance_config = {
            "team": "platform-engineering",
            "project": "auto-instrumentation-demo",
            "environment": "development",
            "cost_center": "engineering-ops",
            "enable_cost_alerts": True,
            "budget_threshold": 5.0,  # $5 daily budget
        }

        # Enable auto-instrumentation - this enhances ALL OpenLLMetry operations
        auto_instrument(**governance_config)

        print("🛡️ Auto-instrumentation configured:")
        print(f"   • Team attribution: {governance_config['team']}")
        print(f"   • Project tracking: {governance_config['project']}")
        print(f"   • Environment: {governance_config['environment']}")
        print(f"   • Budget monitoring: ${governance_config['budget_threshold']}/day")
        print("   • Cost alerts: Enabled")

        return True

    except ImportError as e:
        print(f"❌ Failed to import GenOps auto-instrumentation: {e}")
        print("💡 Fix: Run 'pip install genops[traceloop]'")
        return False
    except Exception as e:
        print(f"❌ Auto-instrumentation setup failed: {e}")
        print("🔧 Setup Troubleshooting:")
        print("   • Verify OpenLLMetry installation: pip list | grep openllmetry")
        print("   • Check GenOps installation: pip install genops[traceloop]")
        print("   • Restart Python interpreter after installation")
        if "import" in str(e).lower():
            print(
                "   💡 Import Error: Missing dependencies - run 'pip install genops[traceloop]'"
            )
        elif "version" in str(e).lower():
            print(
                "   💡 Version Conflict: Update packages - run 'pip install --upgrade genops[traceloop]'"
            )
        return False


def existing_openllmetry_code():
    """
    Simulate existing OpenLLMetry application code.

    This represents code that already exists and uses OpenLLMetry patterns.
    With GenOps auto-instrumentation, this code gets enhanced automatically
    without any modifications.
    """
    print("\n📝 Running Existing OpenLLMetry Application Code")
    print("-" * 45)
    print("ℹ️  Note: This code remains unchanged - GenOps enhancement is automatic")

    try:
        # Standard OpenLLMetry imports and setup
        import openai
        from openllmetry.instrumentation.openai import OpenAIInstrumentor

        # Initialize OpenLLMetry instrumentation (standard pattern)
        OpenAIInstrumentor().instrument()

        client = openai.OpenAI()
        print("✅ Standard OpenLLMetry instrumentation initialized")

    except ImportError as e:
        print(f"❌ OpenLLMetry dependencies missing: {e}")
        print("💡 Fix: Run 'pip install openllmetry'")
        return False

    # Example 1: Standard chat completion (unchanged existing code)
    print("\n1️⃣ Standard Chat Completion (Existing Code)")
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain auto-instrumentation benefits."},
            ],
            max_tokens=100,
        )

        content = response.choices[0].message.content
        print(f"✅ Response: {content[:80]}...")
        print("🛡️ GenOps governance automatically applied:")
        print("   • Cost calculated and attributed to team")
        print("   • Team and project context added to trace")
        print("   • Budget monitoring active")

    except Exception as e:
        print(f"❌ Chat completion failed: {e}")
        return False

    # Example 2: Multiple operations (unchanged existing code)
    print("\n2️⃣ Batch Operations (Existing Code)")
    try:
        prompts = [
            "What is machine learning?",
            "Explain neural networks briefly.",
            "What are transformers in AI?",
        ]

        total_responses = []
        for i, prompt in enumerate(prompts):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
            )

            content = response.choices[0].message.content
            total_responses.append(content)
            print(f"   ✅ Batch item {i + 1}: Generated response")

        print(f"✅ Processed {len(total_responses)} prompts")
        print("🛡️ GenOps automatically provided:")
        print("   • Individual cost tracking for each operation")
        print("   • Batch-level cost aggregation")
        print("   • Team attribution for entire batch")
        print("   • Budget compliance checking")

    except Exception as e:
        print(f"❌ Batch operations failed: {e}")
        return False

    # Example 3: Streaming (unchanged existing code)
    print("\n3️⃣ Streaming Response (Existing Code)")
    try:
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Count from 1 to 5"}],
            max_tokens=50,
            stream=True,
        )

        collected_content = []
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content_piece = chunk.choices[0].delta.content
                collected_content.append(content_piece)

        full_response = "".join(collected_content)
        print(f"✅ Streaming response: {full_response.strip()}")
        print("🛡️ GenOps streaming enhancements:")
        print("   • Real-time cost calculation during streaming")
        print("   • Stream-level governance tracking")
        print("   • Automatic completion cost attribution")

    except Exception as e:
        print(f"❌ Streaming failed: {e}")
        return False

    return True


def demonstrate_governance_transparency():
    """Show how auto-instrumentation provides governance transparency."""
    print("\n👀 Governance Transparency Demo")
    print("-" * 35)

    try:
        from genops.providers.traceloop import get_current_governance_context

        # Get current governance context (added by auto-instrumentation)
        context = get_current_governance_context()

        print("✅ Current governance context:")
        print(f"   • Team: {context.get('team', 'N/A')}")
        print(f"   • Project: {context.get('project', 'N/A')}")
        print(f"   • Environment: {context.get('environment', 'N/A')}")
        print(f"   • Cost center: {context.get('cost_center', 'N/A')}")

        # Show budget status
        from genops.providers.traceloop import get_budget_status

        budget_status = get_budget_status()

        print("\n💰 Budget monitoring status:")
        print(f"   • Daily budget: ${budget_status.get('daily_limit', 'N/A')}")
        print(f"   • Current usage: ${budget_status.get('current_usage', 0.00):.4f}")
        print(f"   • Remaining: ${budget_status.get('remaining', 'N/A')}")

        # Show recent operations summary
        from genops.providers.traceloop import get_recent_operations_summary

        summary = get_recent_operations_summary(limit=5)

        print("\n📊 Recent operations summary:")
        for i, op in enumerate(summary.get("operations", [])):
            print(
                f"   {i + 1}. {op.get('operation_type', 'unknown')}: ${op.get('cost', 0.00):.6f}"
            )

        total_cost = summary.get("total_cost", 0.0)
        print(f"   Total recent cost: ${total_cost:.6f}")

    except Exception as e:
        print(f"❌ Governance transparency demo failed: {e}")
        return False

    return True


def show_migration_benefits():
    """Show benefits of migrating to GenOps-enhanced OpenLLMetry."""
    print("\n🔄 Migration Benefits")
    print("-" * 20)

    print("✅ Zero Code Changes Required:")
    print("   • Keep your existing OpenLLMetry code")
    print(
        "   • Add one line: auto_instrument(team='your-team', project='your-project')"
    )
    print("   • All existing operations get enhanced automatically")

    print("\n💰 Immediate Cost Intelligence:")
    print("   • Automatic cost calculation for all operations")
    print("   • Team and project cost attribution")
    print("   • Real-time budget monitoring and alerts")

    print("\n🛡️ Governance Without Complexity:")
    print("   • Policy enforcement integrated into existing workflows")
    print("   • Compliance tracking for audit requirements")
    print("   • No changes to deployment or infrastructure")

    print("\n🔍 Enhanced Observability:")
    print("   • All existing OpenTelemetry backends work unchanged")
    print("   • Enhanced traces with business context")
    print("   • Governance attributes in every span")

    print("\n🏢 Enterprise Ready:")
    print("   • Scales with your existing OpenLLMetry infrastructure")
    print("   • Optional Traceloop platform integration")
    print("   • Professional support and enterprise features available")


def demonstrate_compatibility():
    """Demonstrate compatibility with existing OpenLLMetry patterns."""
    print("\n🔗 Compatibility Demonstration")
    print("-" * 30)

    try:
        # Show that existing OpenLLMetry patterns still work
        from openllmetry import tracer

        from genops.providers.traceloop import is_enhanced_tracer

        # Check if tracer is enhanced with GenOps
        enhanced = is_enhanced_tracer(tracer)
        print(f"✅ OpenLLMetry tracer enhanced: {enhanced}")

        # Show that manual spans still work with enhancement
        with tracer.start_span("manual_span_example") as span:
            span.set_attribute("user.action", "manual_span_creation")
            span.set_attribute("custom.attribute", "works_as_expected")

            # GenOps automatically adds governance attributes
            print("✅ Manual span created with automatic GenOps enhancement")
            print("   • Original OpenLLMetry attributes preserved")
            print("   • GenOps governance attributes added automatically")
            print("   • Cost tracking enabled for manual spans")

        # Show decorator compatibility
        from openllmetry.decorators import workflow

        @workflow(name="existing_workflow")
        def existing_decorated_function():
            """Existing function with OpenLLMetry decorator."""
            import openai

            client = openai.OpenAI()

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test compatibility"}],
                max_tokens=20,
            )
            return response.choices[0].message.content

        # Execute decorated function - gets both OpenLLMetry and GenOps enhancement
        result = existing_decorated_function()
        print("✅ Existing @workflow decorator enhanced automatically")
        print("   • OpenLLMetry workflow tracking preserved")
        print("   • GenOps governance added seamlessly")
        print(f"   • Result: {result[:50]}...")

    except Exception as e:
        print(f"❌ Compatibility demo failed: {e}")
        return False

    return True


async def main():
    """Main execution function."""
    print("⚡ Auto-Instrumentation OpenLLMetry + GenOps Demo")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found")
        print("💡 Set your OpenAI API key: export OPENAI_API_KEY='your-key'")
        return False

    # Run demo steps
    success = True

    # Set up auto-instrumentation
    if not setup_auto_instrumentation():
        success = False

    # Run existing code (unchanged)
    if success and not existing_openllmetry_code():
        success = False

    # Show governance transparency
    if success and not demonstrate_governance_transparency():
        success = False

    # Show compatibility
    if success and not demonstrate_compatibility():
        success = False

    # Show migration benefits
    if success:
        show_migration_benefits()

    if success:
        print("\n" + "⚡" * 55)
        print("🎉 Auto-Instrumentation Demo Complete!")

        print("\n🚀 What You've Accomplished:")
        print("   ✅ Zero-code enhancement of existing OpenLLMetry applications")
        print("   ✅ Automatic governance for all LLM operations")
        print("   ✅ Cost attribution and budget monitoring")
        print("   ✅ 100% compatibility with existing code")

        print("\n💡 Implementation in Your App:")
        print("   1. Add to your startup code:")
        print("      ```python")
        print("      from genops.providers.traceloop import auto_instrument")
        print("      auto_instrument(team='your-team', project='your-project')")
        print("      ```")
        print("   2. That's it! All existing OpenLLMetry code is enhanced")

        print("\n📊 Immediate Benefits:")
        print("   • 🔍 Enhanced observability with governance context")
        print("   • 💰 Automatic cost calculation and attribution")
        print("   • 🛡️ Policy enforcement and compliance tracking")
        print("   • 📈 Budget monitoring and cost optimization")

        print("\n📚 Next Steps:")
        print("   • Customize governance policies for your organization")
        print("   • Set up budget alerts and approval workflows")
        print("   • Explore Traceloop platform for advanced insights")
        print("   • Integrate with your existing observability stack")

        print("⚡" * 55)
    else:
        print("\n❌ Demo encountered errors. Please check the output above.")

    return success


if __name__ == "__main__":
    asyncio.run(main())
