#!/usr/bin/env python3
"""
⚡ GenOps Gemini Minimal Example - Phase 1 (30 seconds)

This is the absolute simplest way to prove GenOps Gemini integration works.
Perfect for first-time users - instant confidence builder!

Requirements:
- GEMINI_API_KEY environment variable (get free at https://ai.google.dev/)
- pip install google-generativeai genops-ai

Usage:
    python hello_genops_minimal.py

Expected result: "✅ Success! GenOps is now tracking your Gemini usage!"
"""


def main():
    print("🚀 Testing GenOps with Google Gemini...")

    try:
        # Step 1: Enable GenOps tracking (universal CLAUDE.md standard)
        from genops.providers.gemini import auto_instrument

        auto_instrument()
        print("✅ GenOps auto-instrumentation enabled")

        # Step 2: Use Gemini normally - now with GenOps tracking!
        import os

        from google import genai

        # Check for API key with specific guidance
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY environment variable not set")
            print()
            print("🔧 QUICK FIX (copy-paste these commands):")
            print("   1. Get FREE API key: https://ai.google.dev/")
            print("      → Click 'Get API key' → 'Create API key in new project'")
            print("   2. export GEMINI_API_KEY='paste_your_api_key_here'")
            print("   3. python hello_genops_minimal.py")
            print()
            return False

        client = genai.Client(api_key=api_key)

        client.models.generate_content(model="gemini-2.5-flash", contents="Say hello!")

        print("✅ SUCCESS! GenOps is now tracking your Gemini usage!")
        print("💰 Cost tracking, team attribution, and governance are active.")
        print("📊 Your AI operations are now visible in your observability platform.")
        print()
        print("🎯 PHASE 1 COMPLETE - You now have GenOps working!")

        return True

    except ImportError as e:
        if "genai" in str(e):
            print("❌ Google Gemini SDK not installed")
            print("🔧 QUICK FIX: pip install google-generativeai")
        else:
            print("❌ GenOps not installed")
            print("🔧 QUICK FIX: pip install genops-ai[gemini]")
        return False
    except Exception as e:
        error_str = str(e).lower()
        print(f"❌ Error: {e}")
        print()

        # Provide specific guidance for common errors
        if "authentication" in error_str or "api_key" in error_str:
            print("🔧 API KEY ISSUE:")
            print("   1. Check your API key: echo $GEMINI_API_KEY")
            print("   2. Get new key: https://ai.google.dev/")
            print("   3. export GEMINI_API_KEY='your_new_key'")
        elif "quota" in error_str or "rate" in error_str:
            print("🔧 QUOTA/RATE LIMIT:")
            print("   1. Wait 1-2 minutes and try again")
            print("   2. Free tier has limits - upgrade if needed")
        else:
            print("🔧 DETAILED DIAGNOSIS:")
            print(
                '   python -c "from genops.providers.gemini import validate_setup, print_validation_result; print_validation_result(validate_setup(), detailed=True)"'
            )

        return False


if __name__ == "__main__":
    success = main()

    if success:
        print("🚀 READY FOR PHASE 2? (Team Attribution & Control)")
        print("   → python basic_tracking.py        # Add team cost tracking")
        print("   → python auto_instrumentation.py  # Zero-code existing apps")
        print()
        print("📚 Or explore the complete learning path:")
        print("   → examples/gemini/README.md")
    else:
        print()
        print("💡 Need help? Check the troubleshooting guide:")
        print("   → examples/gemini/README.md#troubleshooting")

    exit(0 if success else 1)
