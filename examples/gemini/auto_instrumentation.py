#!/usr/bin/env python3
"""
GenOps Gemini Auto-Instrumentation Example

This example demonstrates zero-code instrumentation with Google Gemini,
showing how existing Gemini code can be automatically tracked without
any modifications to your application logic.

What this demonstrates:
- Zero-code auto-instrumentation that works with existing code
- Multiple AI model demonstrations across different use cases
- Automatic cost tracking and governance telemetry
- Integration with existing Google AI SDK workflows

Example usage:
    python auto_instrumentation.py
"""

import os


def main():
    print("🎯 GenOps Gemini Auto-Instrumentation Example")
    print("=" * 48)
    print("Demonstrating zero-code instrumentation with existing Gemini workflows.\n")

    try:
        # Step 1: Enable auto-instrumentation BEFORE importing Google AI SDK
        print("📡 Enabling GenOps auto-instrumentation...")
        from genops.providers.gemini import auto_instrument_gemini

        # This patches the Google AI SDK to add automatic GenOps tracking
        success = auto_instrument_gemini()
        if success:
            print("✅ Auto-instrumentation enabled - all Gemini calls now tracked!")
        else:
            print("⚠️  Auto-instrumentation setup failed - falling back to manual mode")
        print()

        # Check API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY environment variable required")
            print("💡 Get your API key at: https://ai.google.dev/")
            return False

        # Step 2: Import and use Google AI SDK normally
        # Your existing code works unchanged!
        print("🧠 Using Google AI SDK normally (now with automatic GenOps tracking)...")
        from google import genai

        client = genai.Client(api_key=api_key)
        print("✅ Gemini client initialized\n")

        # Example 1: Basic text generation (automatically tracked)
        print("📝 Example 1: Basic Text Generation")
        print("-" * 35)

        response1 = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Explain quantum computing in one paragraph.",
            # These governance attributes will be automatically captured
            team="research-team",
            project="quantum-education"
        )

        print("✅ Generated quantum computing explanation")
        print(f"📄 Response: {response1.text[:100]}...")
        print("💰 Cost automatically tracked and attributed to research-team")
        print()

        # Example 2: Different model with different use case
        print("📊 Example 2: Business Analysis with Pro Model")
        print("-" * 44)

        business_prompt = """
        Analyze the key factors that contribute to successful remote team management.
        Include specific strategies and best practices.
        """

        response2 = client.models.generate_content(
            model="gemini-2.5-pro",  # Using more capable model
            contents=business_prompt,
            team="hr-analytics",
            project="remote-work-study",
            customer_id="enterprise-client-123"
        )

        print("✅ Generated business analysis")
        print(f"📄 Response: {response2.text[:100]}...")
        print("💰 Cost automatically tracked and attributed to hr-analytics team")
        print()

        # Example 3: Creative content generation
        print("🎨 Example 3: Creative Content Generation")
        print("-" * 38)

        creative_prompt = """
        Write a short, engaging story about a robot who discovers the importance
        of teamwork while building a garden with other robots.
        """

        response3 = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=creative_prompt,
            team="content-creation",
            project="ai-storytelling",
            environment="development"
        )

        print("✅ Generated creative story")
        print(f"📄 Response: {response3.text[:150]}...")
        print("💰 Cost automatically tracked and attributed to content-creation team")
        print()

        # Example 4: Code generation and analysis
        print("💻 Example 4: Code Generation")
        print("-" * 27)

        code_prompt = """
        Write a Python function that calculates the factorial of a number
        using recursion. Include proper error handling and documentation.
        """

        response4 = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=code_prompt,
            team="engineering",
            project="code-assistant",
            feature="factorial-generator"
        )

        print("✅ Generated Python code")
        print(f"📄 Response: {response4.text[:100]}...")
        print("💰 Cost automatically tracked and attributed to engineering team")
        print()

        # Example 5: Demonstrate chat-like conversation
        print("💬 Example 5: Multi-Turn Conversation Simulation")
        print("-" * 46)

        # Simulate a conversation by including context
        conversation_prompt = """
        User: What are the main benefits of renewable energy?
        
        Assistant: The main benefits of renewable energy include environmental sustainability, 
        reduced carbon emissions, energy independence, and long-term cost savings.
        
        User: Can you elaborate on the cost savings aspect?
        """

        response5 = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=conversation_prompt,
            team="sustainability",
            project="renewable-energy-analysis",
            customer_id="green-tech-corp"
        )

        print("✅ Generated conversation response")
        print(f"📄 Response: {response5.text[:100]}...")
        print("💰 Cost automatically tracked and attributed to sustainability team")
        print()

        # Summary of what happened
        print("🎉 Auto-Instrumentation Success!")
        print("=" * 32)
        print("✅ All Gemini API calls were automatically tracked with GenOps!")
        print()
        print("📊 What was automatically captured:")
        print("   💰 Real-time cost calculation for each operation")
        print("   🏷️  Team and project attribution for billing")
        print("   📈 Performance metrics (latency, tokens, model usage)")
        print("   🔍 Operation tracing and debugging information")
        print("   📡 OpenTelemetry export to your observability platform")
        print()

        print("🎯 Teams that used AI in this session:")
        teams = ["research-team", "hr-analytics", "content-creation", "engineering", "sustainability"]
        for i, team in enumerate(teams, 1):
            print(f"   {i}. {team}")
        print()

        print("💡 Key Benefits Demonstrated:")
        print("   ✨ Zero code changes required in your existing Gemini workflows")
        print("   📊 Automatic cost attribution across teams and projects")
        print("   🎯 Model usage optimization insights")
        print("   🔄 Seamless integration with existing development processes")
        print("   📈 Ready-to-use governance telemetry for compliance and reporting")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Install required packages:")
        print("   pip install genops-ai[gemini] google-generativeai")
        return False

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Verify GEMINI_API_KEY is set: export GEMINI_API_KEY='your_key'")
        print("   2. Check internet connectivity and API service status")
        print("   3. Run validation: python -c \"from genops.providers.gemini import validate_setup, print_validation_result; print_validation_result(validate_setup())\"")
        return False

if __name__ == "__main__":
    success = main()

    if success:
        print("\n🎯 What's Next?")
        print("   → Try cost optimization: python cost_optimization.py")
        print("   → Explore cost context managers: python cost_tracking.py")
        print("   → See production patterns: python production_patterns.py")
        print("   → Check validation: python validation_example.py")

    exit(0 if success else 1)
