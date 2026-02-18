#!/usr/bin/env python3
"""
OpenAI Auto-Instrumentation Example

This example demonstrates GenOps zero-code auto-instrumentation for OpenAI.
Your existing OpenAI code works unchanged, but gets automatic governance telemetry.

What you'll learn:
- Zero-code setup with auto_instrument()
- Governance context for cost attribution
- Transparent telemetry with no API changes

Usage:
    python auto_instrumentation.py

Prerequisites:
    pip install genops-ai[openai]
    export OPENAI_API_KEY="your_api_key_here"
"""

import os
import sys


def setup_auto_instrumentation():
    """Set up GenOps auto-instrumentation for OpenAI."""
    print("🔧 Setting Up Auto-Instrumentation")
    print("-" * 40)

    try:
        # This single line enables automatic telemetry for ALL OpenAI operations
        from genops import auto_instrument

        auto_instrument()

        print("✅ GenOps auto-instrumentation enabled!")
        print("   • All OpenAI operations will automatically include telemetry")
        print("   • No changes to your existing OpenAI code required")
        print("   • Cost and performance data automatically captured")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Fix: Run 'pip install genops-ai[openai]'")
        return False


def existing_openai_code_unchanged():
    """Your existing OpenAI code works exactly as before, but with automatic telemetry."""
    print("\n\n💻 Your Existing OpenAI Code (Unchanged)")
    print("-" * 50)

    try:
        # This is your normal OpenAI code - no changes needed!
        from openai import OpenAI

        client = OpenAI()  # Uses OPENAI_API_KEY from environment

        print("🚀 Making standard OpenAI requests...")

        # Example 1: Simple chat completion (your existing code)
        response1 = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "What is machine learning?"}],
            max_tokens=100,
        )

        print(f"✅ Response 1: {response1.choices[0].message.content[:50]}...")

        # Example 2: More complex completion (your existing code)
        response2 = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful data scientist."},
                {"role": "user", "content": "Explain the bias-variance tradeoff"},
            ],
            temperature=0.7,
            max_tokens=200,
        )

        print(f"✅ Response 2: {response2.choices[0].message.content[:50]}...")

        # Example 3: Legacy completion endpoint (if you use it)
        try:
            response3 = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt="Write a haiku about programming:",
                max_tokens=50,
            )
            print(f"✅ Response 3: {response3.choices[0].text.strip()[:50]}...")
        except Exception as e:
            print(f"ℹ️  Legacy completions skipped: {e}")

        print("\n🎯 Key Point: Zero code changes, automatic telemetry!")
        print("   • All requests above were automatically tracked")
        print("   • Cost calculations performed automatically")
        print("   • Performance metrics captured automatically")

        return True

    except Exception as e:
        print(f"❌ Error with existing OpenAI code: {e}")
        print("💡 Check your OPENAI_API_KEY and network connectivity")
        return False


def add_governance_context():
    """Add governance context to automatically apply to all operations."""
    print("\n\n🏷️  Adding Governance Context")
    print("-" * 40)

    try:
        from openai import OpenAI

        from genops.core.context import set_governance_context

        # Set governance context once - applies to ALL subsequent operations
        set_governance_context(
            {
                "team": "auto-instrumentation-demo",
                "project": "genops-examples",
                "customer_id": "demo-customer-auto",
                "environment": "development",
                "cost_center": "engineering-dept",
            }
        )

        print("✅ Governance context set for all operations:")
        print("   • team: auto-instrumentation-demo")
        print("   • project: genops-examples")
        print("   • customer_id: demo-customer-auto")
        print("   • environment: development")

        # Now all OpenAI operations automatically inherit these attributes
        client = OpenAI()

        print("\n🚀 Making requests with automatic governance attribution...")

        # These requests automatically get the governance context above
        tasks = [
            "Explain quantum computing briefly",
            "What are the benefits of renewable energy?",
            "How do neural networks learn?",
        ]

        for i, task in enumerate(tasks, 1):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": task}],
                max_tokens=50,
            )

            print(f"   {i}. Task: {task}")
            print(f"      Response: {response.choices[0].message.content[:40]}...")

        print("\n💰 All costs automatically attributed to:")
        print("   • Team: auto-instrumentation-demo")
        print("   • Project: genops-examples")
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
        from openai import OpenAI

        from genops.core.context import set_governance_context

        # Simulate web application request handler
        def handle_chat_request(user_id: str, message: str, session_id: str):
            """Simulated web app request handler with automatic telemetry."""

            # Set request-specific governance context
            set_governance_context(
                {
                    "team": "web-app-team",
                    "project": "customer-chat-api",
                    "customer_id": user_id,
                    "environment": "production",
                    "feature": "chat-endpoint",
                    "session_id": session_id,
                }
            )

            # Your normal OpenAI code - completely unchanged
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful customer service assistant.",
                    },
                    {"role": "user", "content": message},
                ],
                max_tokens=150,
            )

            return response.choices[0].message.content

        # Simulate multiple user requests
        print("🔄 Simulating web application requests...")

        simulated_requests = [
            ("user-001", "How do I reset my password?", "session-abc-123"),
            ("user-002", "What are your business hours?", "session-def-456"),
            ("user-003", "I need help with billing", "session-ghi-789"),
        ]

        for user_id, message, session_id in simulated_requests:
            response = handle_chat_request(user_id, message, session_id)
            print(f"   User {user_id}: {message}")
            print(f"   Response: {response[:60]}...")
            print()

        print("✅ Web application pattern complete!")
        print("💡 Each request automatically gets:")
        print("   • User-specific cost attribution")
        print("   • Session tracking")
        print("   • Feature-level cost allocation")
        print("   • Environment and team attribution")

        return True

    except Exception as e:
        print(f"❌ Web application pattern error: {e}")
        return False


def main():
    """Run auto-instrumentation demonstration."""
    print("🤖 GenOps OpenAI Auto-Instrumentation Demo")
    print("=" * 60)

    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("💡 Fix: export OPENAI_API_KEY='your_api_key_here'")
        return False

    success = True

    # Run demonstrations
    success &= setup_auto_instrumentation()
    success &= existing_openai_code_unchanged()
    success &= add_governance_context()
    success &= web_application_pattern()

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("🎉 Auto-instrumentation demonstration complete!")

        print("\n🔑 Key Takeaways:")
        print("   ✅ One line enables telemetry: auto_instrument()")
        print("   ✅ Zero changes to existing OpenAI code")
        print("   ✅ Automatic cost calculation and attribution")
        print("   ✅ Governance context applies to all operations")
        print("   ✅ Perfect for web applications and microservices")

        print("\n💰 Benefits:")
        print("   • Instant cost visibility across all OpenAI usage")
        print("   • Automatic attribution to teams, projects, customers")
        print("   • No code refactoring or API changes required")
        print("   • Drop-in replacement for existing applications")

        print("\n🚀 Next Steps:")
        print("   • Run 'python cost_optimization.py' for multi-model strategies")
        print("   • Try 'python advanced_features.py' for streaming and functions")
        print("   • Explore 'python production_patterns.py' for enterprise patterns")

        return True
    else:
        print("❌ Auto-instrumentation demonstration failed.")
        print("💡 Check the error messages above and try setup_validation.py")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
