#!/usr/bin/env python3
"""
⚡ GenOps Replicate Minimal Example - Phase 1 (30 seconds)

This is the absolute simplest way to prove GenOps Replicate integration works.
Perfect for first-time users - instant confidence builder!

Requirements: 
- REPLICATE_API_TOKEN environment variable (get free at https://replicate.com/account/api-tokens)
- pip install replicate genops-ai

Usage:
    python hello_genops_minimal.py
    
Expected result: "✅ SUCCESS! GenOps is now tracking your Replicate usage!"
"""

def main():
    print("🚀 Testing GenOps with Replicate...")

    try:
        # Step 1: Enable GenOps tracking (universal CLAUDE.md standard)
        from genops.providers.replicate import auto_instrument
        auto_instrument()
        print("✅ GenOps auto-instrumentation enabled")

        # Step 2: Use Replicate normally - now with GenOps tracking!
        import os

        import replicate

        # Check for API token with specific guidance
        api_token = os.getenv("REPLICATE_API_TOKEN")
        if not api_token:
            print("❌ REPLICATE_API_TOKEN environment variable not set")
            print()
            print("🔧 QUICK FIX (copy-paste these commands):")
            print("   1. Get FREE API token: https://replicate.com/account/api-tokens")
            print("      → Sign up/log in → Click 'Create token'")
            print("   2. export REPLICATE_API_TOKEN='r8_paste_your_token_here'")
            print("   3. python hello_genops_minimal.py")
            print()
            return False

        # Simple test with a fast, cheap model
        print("🤖 Running test with Replicate model...")
        output = replicate.run(
            "meta/llama-2-7b-chat",
            input={
                "prompt": "Say hello!",
                "max_length": 50,
                "temperature": 0.7
            }
        )

        print("✅ SUCCESS! GenOps is now tracking your Replicate usage!")
        print("💰 Cost tracking, team attribution, and governance are active.")
        print("📊 Your AI operations are now visible in your observability platform.")
        print()
        print(f"🤖 Model response: {output[:100] if output else 'Success'}...")
        print()
        print("🎯 PHASE 1 COMPLETE - You now have GenOps working with Replicate!")

        return True

    except ImportError as e:
        if "replicate" in str(e):
            print("❌ Replicate SDK not installed")
            print("🔧 QUICK FIX: pip install replicate")
        else:
            print("❌ GenOps not installed")
            print("🔧 QUICK FIX: pip install genops-ai[replicate]")
        return False
    except Exception as e:
        error_str = str(e).lower()
        print(f"❌ Error: {e}")
        print()

        # Provide specific guidance for common errors
        if "authentication" in error_str or "token" in error_str:
            print("🔧 API TOKEN ISSUE:")
            print("   1. Check your token: echo $REPLICATE_API_TOKEN")
            print("   2. Get new token: https://replicate.com/account/api-tokens")
            print("   3. export REPLICATE_API_TOKEN='r8_your_new_token'")
        elif "model not found" in error_str or "404" in error_str:
            print("🔧 MODEL AVAILABILITY:")
            print("   1. Try a different model from: https://replicate.com/explore")
            print("   2. Check model name spelling and format")
        elif "rate limit" in error_str or "quota" in error_str:
            print("🔧 RATE LIMIT:")
            print("   1. Wait 1-2 minutes and try again")
            print("   2. Free tier has usage limits")
        else:
            print("🔧 DETAILED DIAGNOSIS:")
            print("   python -c \"from genops.providers.replicate_validation import validate_setup, print_validation_result; print_validation_result(validate_setup(), detailed=True)\"")

        return False

if __name__ == "__main__":
    success = main()

    if success:
        print("🚀 READY FOR PHASE 2? (Team Attribution & Multi-Modal)")
        print("   → python basic_tracking.py          # Add team cost tracking")
        print("   → python auto_instrumentation.py    # Zero-code existing apps")
        print()
        print("📚 Or explore the complete learning path:")
        print("   → examples/replicate/README.md")
    else:
        print()
        print("💡 Need help? Check the troubleshooting guide:")
        print("   → examples/replicate/README.md#troubleshooting")

    exit(0 if success else 1)
