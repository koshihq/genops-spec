#!/usr/bin/env python3
"""
GenOps Gemini Hello World Example

This example demonstrates GenOps integration with Google Gemini AI, showing
automatic cost tracking, governance telemetry, and basic usage patterns.

What this demonstrates:
- Zero-code instrumentation setup with Gemini
- Basic AI operation with automatic governance
- Immediate confirmation that GenOps is working with Google Gemini
- API key validation and setup verification

Example usage:
    python hello_genops.py
"""

import os
import sys


def main():
    """Comprehensive GenOps Gemini example with detailed guidance."""

    print("👋 GenOps Gemini Hello World Example")
    print("=" * 40)
    print("This example shows GenOps cost tracking and governance with Google Gemini.")
    print()

    try:
        # Step 1: Enable GenOps instrumentation for Gemini
        print("📡 Enabling GenOps Gemini instrumentation...")
        from genops.providers.gemini import instrument_gemini
        instrument_gemini()
        print("✅ GenOps Gemini instrumentation enabled!")

        # Step 2: Verify API key configuration
        print("\n🔐 Checking API key configuration...")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY environment variable not set")
            print("\n💡 To fix this:")
            print("   1. Get your API key from: https://ai.google.dev/")
            print("   2. Set environment variable: export GEMINI_API_KEY='your_api_key_here'")
            print("   3. Re-run this example")
            return False

        print("✅ API key found and configured!")

        # Step 3: Use Gemini normally with Google AI SDK
        print("\n🧠 Making Google Gemini API call...")
        from google import genai

        client = genai.Client(api_key=api_key)

        # This single call now has comprehensive AI governance!
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Hello from GenOps! Please respond with a friendly greeting."
        )

        # Extract response text
        ai_response = response.text if hasattr(response, 'text') else str(response)

        # Step 4: Celebrate success!
        print("✅ Success! AI operation completed with GenOps governance!")
        print(f"🤖 Gemini Response: {ai_response.strip()}")
        print()
        print("🎉 Congratulations! GenOps is now tracking:")
        print("   💰 Real-time cost calculation with token-level precision")
        print("   🏛️  Governance and compliance data with audit trails")
        print("   📊 Performance and usage metrics with model comparisons")
        print("   🔍 Error tracking and debugging information")
        print("   📡 OpenTelemetry export to your observability platform")
        print()
        print("🚀 You're ready to explore more advanced GenOps Gemini features!")

        return True

    except ImportError as e:
        error_str = str(e)
        print(f"❌ Import error: {error_str}")
        print("\n💡 Fix this by installing required packages:")

        if "genai" in error_str:
            print("   pip install google-generativeai")
        if "genops" in error_str:
            print("   pip install genops-ai[gemini]")

        print("\n   # Or install both:")
        print("   pip install genops-ai[gemini] google-generativeai")
        return False

    except Exception as e:
        error_str = str(e)
        print(f"❌ Error: {error_str}")
        print(f"   Error type: {type(e).__name__}")
        print("\n💡 Common fixes:")

        if "api" in error_str.lower() or "key" in error_str.lower():
            print("   - Verify GEMINI_API_KEY environment variable is set correctly")
            print("   - Check that your API key is valid and active")
            print("   - Ensure API key has proper permissions")
            print("   - Get a new API key from: https://ai.google.dev/")
        elif "quota" in error_str.lower() or "limit" in error_str.lower():
            print("   - API quota or rate limit exceeded")
            print("   - Wait a few minutes and try again")
            print("   - Consider upgrading to paid tier for higher limits")
        elif "network" in error_str.lower() or "connection" in error_str.lower():
            print("   - Check your internet connection")
            print("   - Verify Gemini API service is accessible")
            print("   - Try again in a few minutes")
        else:
            print("   - Run validation script: python -c \"from genops.providers.gemini import validate_setup, print_validation_result; print_validation_result(validate_setup(), detailed=True)\"")
            print("   - Check Google AI service status")
            print("   - Verify your API key and permissions")

        return False

if __name__ == "__main__":
    success = main()

    if success:
        print("\n🎯 What's Next?")
        print("   1. Try: python basic_tracking.py")
        print("   2. Explore: python cost_optimization.py")
        print("   3. Advanced: python auto_instrumentation.py")
        print("   4. Production: python production_patterns.py")
        print("\n📖 Learn More:")
        print("   → Quickstart: docs/gemini-quickstart.md")
        print("   → Full Guide: docs/integrations/gemini.md")

    sys.exit(0 if success else 1)
