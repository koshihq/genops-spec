#!/usr/bin/env python3
"""
Anthropic Auto-Instrumentation Example

This example demonstrates GenOps zero-code auto-instrumentation for Anthropic Claude.
Your existing Anthropic code works unchanged, but gets automatic governance telemetry.

What you'll learn:
- Zero-code setup with auto_instrument()
- Governance context for cost attribution
- Transparent telemetry with no API changes

Usage:
    python auto_instrumentation.py

Prerequisites:
    pip install genops-ai[anthropic]
    export ANTHROPIC_API_KEY="your_anthropic_key_here"
"""

import os
import sys


def setup_auto_instrumentation():
    """Set up GenOps auto-instrumentation for Anthropic."""
    print("🔧 Setting Up Auto-Instrumentation")
    print("-" * 40)

    try:
        # This single line enables automatic telemetry for ALL Anthropic operations
        from genops import auto_instrument
        auto_instrument()

        print("✅ GenOps auto-instrumentation enabled!")
        print("   • All Anthropic operations will automatically include telemetry")
        print("   • No changes to your existing Anthropic code required")
        print("   • Cost and performance data automatically captured")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Fix: Run 'pip install genops-ai[anthropic]'")
        return False

def existing_anthropic_code_unchanged():
    """Your existing Anthropic code works exactly as before, but with automatic telemetry."""
    print("\n\n💻 Your Existing Anthropic Code (Unchanged)")
    print("-" * 50)

    try:
        # This is your normal Anthropic code - no changes needed!
        from anthropic import Anthropic

        client = Anthropic()  # Uses ANTHROPIC_API_KEY from environment

        print("🚀 Making standard Anthropic requests...")

        # Example 1: Simple message creation (your existing code)
        response1 = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "What is artificial intelligence?"}]
        )

        print(f"✅ Response 1: {response1.content[0].text[:50]}...")

        # Example 2: More complex message (your existing code)
        response2 = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            messages=[
                {"role": "user", "content": "Explain the benefits and challenges of renewable energy adoption"}
            ],
            temperature=0.7
        )

        print(f"✅ Response 2: {response2.content[0].text[:50]}...")

        # Example 3: System message usage (your existing code)
        response3 = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=150,
            system="You are a helpful coding assistant. Provide clear, concise explanations.",
            messages=[{"role": "user", "content": "What is the difference between a list and a tuple in Python?"}]
        )

        print(f"✅ Response 3: {response3.content[0].text[:50]}...")

        print("\n🎯 Key Point: Zero code changes, automatic telemetry!")
        print("   • All requests above were automatically tracked")
        print("   • Cost calculations performed automatically")
        print("   • Performance metrics captured automatically")

        return True

    except Exception as e:
        print(f"❌ Error with existing Anthropic code: {e}")
        print("💡 Check your ANTHROPIC_API_KEY and network connectivity")
        return False

def add_governance_context():
    """Add governance context to automatically apply to all operations."""
    print("\n\n🏷️  Adding Governance Context")
    print("-" * 40)

    try:
        from anthropic import Anthropic

        from genops.core.context import set_governance_context

        # Set governance context once - applies to ALL subsequent operations
        set_governance_context({
            "team": "auto-instrumentation-demo",
            "project": "genops-anthropic-examples",
            "customer_id": "demo-customer-auto",
            "environment": "development",
            "cost_center": "ai-research-dept"
        })

        print("✅ Governance context set for all operations:")
        print("   • team: auto-instrumentation-demo")
        print("   • project: genops-anthropic-examples")
        print("   • customer_id: demo-customer-auto")
        print("   • environment: development")

        # Now all Anthropic operations automatically inherit these attributes
        client = Anthropic()

        print("\n🚀 Making requests with automatic governance attribution...")

        # These requests automatically get the governance context above
        tasks = [
            "Explain quantum computing in simple terms",
            "What are the advantages of using Claude for content generation?",
            "How can AI help with document analysis?"
        ]

        for i, task in enumerate(tasks, 1):
            response = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=80,
                messages=[{"role": "user", "content": task}]
            )

            print(f"   {i}. Task: {task}")
            print(f"      Response: {response.content[0].text[:60]}...")

        print("\n💰 All costs automatically attributed to:")
        print("   • Team: auto-instrumentation-demo")
        print("   • Project: genops-anthropic-examples")
        print("   • Customer: demo-customer-auto")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error setting governance context: {e}")
        return False

def web_application_pattern():
    """Demonstrate auto-instrumentation in web application context."""
    print("\n\n🌐 Web Application Integration Pattern")
    print("-" * 50)

    try:
        from anthropic import Anthropic

        from genops.core.context import set_governance_context

        # Simulate web application request handler
        def handle_document_analysis(user_id: str, document_type: str, content: str, session_id: str):
            """Simulated web app document analysis handler with automatic telemetry."""

            # Set request-specific governance context
            set_governance_context({
                "team": "document-analysis-team",
                "project": "ai-document-processor",
                "customer_id": user_id,
                "environment": "production",
                "feature": "document-analysis-api",
                "session_id": session_id,
                "document_type": document_type
            })

            # Your normal Anthropic code - completely unchanged
            client = Anthropic()
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Good for analysis
                max_tokens=200,
                system="You are an expert document analyst. Provide clear, structured analysis.",
                messages=[{"role": "user", "content": f"Analyze this {document_type}: {content}"}]
            )

            return response.content[0].text

        # Simulate multiple user requests
        print("🔄 Simulating web application requests...")

        simulated_requests = [
            ("user-001", "contract", "Software license agreement with standard terms", "session-abc-123"),
            ("user-002", "email", "Customer complaint about delayed delivery and refund request", "session-def-456"),
            ("user-003", "report", "Quarterly sales data showing 15% growth in renewable energy sector", "session-ghi-789")
        ]

        for user_id, doc_type, content, session_id in simulated_requests:
            analysis = handle_document_analysis(user_id, doc_type, content, session_id)
            print(f"   User {user_id} ({doc_type}): {content[:40]}...")
            print(f"   Analysis: {analysis[:80]}...")
            print()

        print("✅ Web application pattern complete!")
        print("💡 Each request automatically gets:")
        print("   • User-specific cost attribution")
        print("   • Document type classification")
        print("   • Session and feature-level tracking")
        print("   • Environment and team attribution")

        return True

    except Exception as e:
        print(f"❌ Web application pattern error: {e}")
        return False

def conversational_ai_pattern():
    """Demonstrate auto-instrumentation for conversational AI applications."""
    print("\n\n💬 Conversational AI Pattern")
    print("-" * 50)

    try:
        from anthropic import Anthropic

        from genops.core.context import set_governance_context

        # Simulate a multi-turn conversation
        conversation_history = [
            {"role": "user", "content": "I'm planning a trip to Japan. What should I know?"},
        ]

        # Set conversation-specific context
        set_governance_context({
            "team": "conversational-ai-team",
            "project": "travel-assistant-bot",
            "customer_id": "travel-user-001",
            "environment": "production",
            "feature": "travel-planning",
            "conversation_type": "travel_assistance"
        })

        client = Anthropic()

        print("🗣️  Multi-turn conversation simulation:")

        # Turn 1: Initial response
        print(f"   User: {conversation_history[0]['content']}")

        response1 = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            system="You are a helpful travel assistant. Provide useful, practical advice.",
            messages=conversation_history
        )

        assistant_response1 = response1.content[0].text
        print(f"   Claude: {assistant_response1[:100]}...")

        # Add to conversation history
        conversation_history.append({"role": "assistant", "content": assistant_response1})
        conversation_history.append({"role": "user", "content": "What about the best time to visit?"})

        # Turn 2: Follow-up response
        print("\n   User: What about the best time to visit?")

        response2 = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=150,
            system="You are a helpful travel assistant. Provide useful, practical advice.",
            messages=conversation_history
        )

        assistant_response2 = response2.content[0].text
        print(f"   Claude: {assistant_response2[:100]}...")

        # Add final exchange
        conversation_history.append({"role": "assistant", "content": assistant_response2})
        conversation_history.append({"role": "user", "content": "Thank you! Any cultural tips?"})

        # Turn 3: Cultural advice
        print("\n   User: Thank you! Any cultural tips?")

        response3 = client.messages.create(
            model="claude-3-5-haiku-20241022",  # Faster for final response
            max_tokens=120,
            system="You are a helpful travel assistant. Provide useful, practical advice.",
            messages=conversation_history
        )

        assistant_response3 = response3.content[0].text
        print(f"   Claude: {assistant_response3[:100]}...")

        print("\n💡 Conversation Tracking Benefits:")
        print("   • Each turn automatically tracked with conversation context")
        print("   • Cost attribution across entire conversation session")
        print("   • Model selection optimization per conversation turn")
        print("   • User journey and engagement analytics")

        return True

    except Exception as e:
        print(f"❌ Conversational AI pattern error: {e}")
        return False

def main():
    """Run auto-instrumentation demonstration."""
    print("🤖 GenOps Anthropic Auto-Instrumentation Demo")
    print("=" * 60)

    # Check prerequisites
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY environment variable not set")
        print("💡 Fix: export ANTHROPIC_API_KEY='your_api_key_here'")
        return False

    success = True

    # Run demonstrations
    success &= setup_auto_instrumentation()
    success &= existing_anthropic_code_unchanged()
    success &= add_governance_context()
    success &= web_application_pattern()
    success &= conversational_ai_pattern()

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("🎉 Auto-instrumentation demonstration complete!")

        print("\n🔑 Key Takeaways:")
        print("   ✅ One line enables telemetry: auto_instrument()")
        print("   ✅ Zero changes to existing Anthropic code")
        print("   ✅ Automatic cost calculation and attribution")
        print("   ✅ Governance context applies to all operations")
        print("   ✅ Perfect for web applications and conversational AI")

        print("\n💰 Benefits:")
        print("   • Instant cost visibility across all Claude usage")
        print("   • Automatic attribution to teams, projects, customers")
        print("   • No code refactoring or API changes required")
        print("   • Drop-in replacement for existing applications")
        print("   • Advanced conversation and document analysis tracking")

        print("\n🚀 Next Steps:")
        print("   • Run 'python cost_optimization.py' for Claude model strategies")
        print("   • Try 'python advanced_features.py' for streaming and document analysis")
        print("   • Explore 'python production_patterns.py' for enterprise patterns")

        return True
    else:
        print("❌ Auto-instrumentation demonstration failed.")
        print("💡 Check the error messages above and try setup_validation.py")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
