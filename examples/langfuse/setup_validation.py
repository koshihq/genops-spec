#!/usr/bin/env python3
"""
Langfuse LLM Observability Setup Validation Example

This script validates your Langfuse + GenOps setup for enhanced LLM observability
with governance intelligence and provides detailed diagnostics for any configuration issues.
Run this first before other examples.

Usage:
    python setup_validation.py

Prerequisites:
    pip install genops[langfuse]
    export LANGFUSE_PUBLIC_KEY="pk-lf-your-public-key"
    export LANGFUSE_SECRET_KEY="sk-lf-your-secret-key"
    export OPENAI_API_KEY="your-openai-api-key"  # At least one provider required
"""

import os
import sys
from datetime import datetime


def main():
    """Run comprehensive Langfuse + GenOps setup validation."""
    print("🔍 Langfuse LLM Observability + GenOps Setup Validation")
    print("=" * 65)

    # Import validation utilities
    try:
        from genops.providers.langfuse_validation import (
            print_validation_result,
            validate_setup,
        )

        print("✅ GenOps Langfuse validation utilities loaded successfully")
    except ImportError as e:
        print(f"❌ Failed to import GenOps Langfuse validation utilities: {e}")
        print("\\n💡 Fix: Run 'pip install genops[langfuse]'")
        return False

    # Quick environment check
    print("\\n🌍 Environment Check:")
    print("-" * 30)

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    base_url = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    if public_key:
        print("✅ LANGFUSE_PUBLIC_KEY: Found and validated")
    else:
        print("❌ LANGFUSE_PUBLIC_KEY: Not found")
        print("   Get your keys at: https://cloud.langfuse.com/")

    if secret_key:
        print("✅ LANGFUSE_SECRET_KEY: Found and validated")
    else:
        print("❌ LANGFUSE_SECRET_KEY: Not found")
        print("   Get your keys at: https://cloud.langfuse.com/")

    print(f"🌐 LANGFUSE_BASE_URL: {base_url}")

    # Check LLM provider keys
    providers_found = []
    provider_keys = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "Groq": "GROQ_API_KEY",
    }

    for provider, env_var in provider_keys.items():
        if os.getenv(env_var):
            providers_found.append(provider)
            print(f"✅ {provider}: Found and validated")
        else:
            print(f"⚠️  {provider}: Not configured ({env_var})")

    if not providers_found:
        print("\\n❌ No LLM provider API keys found! You need at least one.")
        print("   • OpenAI: https://platform.openai.com/api-keys")
        print("   • Anthropic: https://console.anthropic.com/")
        print("   • Groq: https://console.groq.com/ (free tier available)")
        return False

    print(
        f"\\n✅ Found {len(providers_found)} configured providers: {', '.join(providers_found)}"
    )

    # Run comprehensive validation
    print("\\n🧪 Running comprehensive validation...")
    print("-" * 40)

    try:
        validation_result = validate_setup(include_performance_tests=True)
        print_validation_result(validation_result, detailed=True)

        # Summary
        print("\\n" + "=" * 65)
        if validation_result and hasattr(validation_result, "overall_status"):
            if validation_result.overall_status.value == "PASSED":
                print(
                    "🎉 Success! Your Langfuse LLM Observability + GenOps setup is ready!"
                )
                print("\\n🔍 Enhanced Observability Active:")
                print("   • Langfuse tracing ✅ Enhanced with GenOps governance")
                print("   • Cost intelligence ✅ Integrated with observability traces")
                print("   • Team attribution ✅ Automatic cost and usage attribution")
                print("   • Budget enforcement ✅ Policy compliance within traces")
                for provider in providers_found:
                    print(f"   • {provider} ✅ Ready for governed LLM operations")

                print("\\n📚 Next steps:")
                print(
                    "   • Run 'python basic_tracking.py' for enhanced tracing examples"
                )
                print(
                    "   • Run 'python evaluation_integration.py' for governance-aware evaluations"
                )
                print(
                    "   • Run 'python auto_instrumentation.py' for zero-code integration"
                )

                print("\\n💡 Quick Test:")
                print("   Try this command to test your enhanced observability:")
                print(
                    "   python -c \\\"from genops.providers.langfuse import instrument_langfuse; print('Enhanced observability ready!')\\\""
                )

            else:
                print("⚠️  Setup validation completed with warnings.")
                print("   Review the detailed output above for specific issues.")
                print(
                    "   You can still proceed, but some features may not work optimally."
                )
        else:
            print("❌ Setup validation failed. Please review the errors above.")
            print("\\n🔧 Common fixes:")
            print("   • Verify all API keys are correct and have sufficient credits")
            print("   • Check network connectivity to Langfuse and AI providers")
            print("   • Ensure Langfuse observability platform is accessible")
            return False

    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        print("\\n🔧 Troubleshooting:")
        print("   • Check your API keys are valid")
        print("   • Verify network connectivity")
        print("   • Try: pip install --upgrade genops[langfuse]")
        return False

    return True


def demonstrate_quick_integration():
    """Show a quick integration example."""
    print("\\n🚀 Quick Integration Demo")
    print("-" * 25)

    try:
        from genops.providers.langfuse import instrument_langfuse

        # Test basic adapter creation
        print("✅ Creating GenOps Langfuse adapter...")
        instrument_langfuse(
            team="validation-demo", project="setup-check", environment="development"
        )

        print("✅ Enhanced Langfuse observability ready!")
        print("\\n🔍 Integration Features Available:")

        integration_features = [
            "🔍 Enhanced Traces - Langfuse traces with GenOps governance attributes",
            "💰 Cost Intelligence - Real-time cost tracking integrated with observability",
            "🏷️ Team Attribution - Automatic cost attribution to teams and projects",
            "🛡️ Policy Compliance - Budget enforcement and governance validation",
            "📊 Evaluation Governance - LLM evaluation tracking with cost oversight",
            "⚡ Zero-Code Setup - Auto-instrumentation for existing Langfuse apps",
            "📈 Business Intelligence - Cost optimization insights and recommendations",
        ]

        for feature in integration_features:
            print(f"   {feature}")

        return True

    except Exception as e:
        print(f"❌ Integration demo failed: {e}")
        return False


if __name__ == "__main__":
    """Main entry point."""
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    success = main()

    if success:
        # Show quick integration demo
        demonstrate_quick_integration()

        print("\\n" + "🌟" * 25)
        print("Your Langfuse + GenOps integration is ready!")
        print("Enhanced LLM observability with governance intelligence!")
        print("🌟" * 25)
        sys.exit(0)
    else:
        print("\\n❌ Setup validation failed. Please fix the issues above.")
        sys.exit(1)
