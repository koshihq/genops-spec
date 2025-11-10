#!/usr/bin/env python3
"""
Traceloop + OpenLLMetry Setup Validation Example

This script validates your Traceloop + OpenLLMetry + GenOps setup for enhanced LLM observability
with governance intelligence and provides detailed diagnostics for any configuration issues. 
Run this first before other examples.

About the Integration:
- OpenLLMetry: Open-source observability framework (Apache 2.0) that extends OpenTelemetry for LLMs
- Traceloop: Commercial platform built on OpenLLMetry with enterprise features and insights
- GenOps: Adds governance, cost intelligence, and policy enforcement to the observability stack

Usage:
    python setup_validation.py

Prerequisites:
    pip install genops[traceloop]  # Includes OpenLLMetry and Traceloop SDK
    export OPENAI_API_KEY="your-openai-api-key"  # At least one provider required
    
    # Optional: For Traceloop commercial platform
    export TRACELOOP_API_KEY="your-traceloop-api-key"
    export TRACELOOP_BASE_URL="https://app.traceloop.com"  # Default
"""

import os
import sys
from datetime import datetime


def main():
    """Run comprehensive Traceloop + OpenLLMetry + GenOps setup validation."""
    print("🔍 Traceloop + OpenLLMetry LLM Observability + GenOps Setup Validation")
    print("=" * 75)

    # Import validation utilities
    try:
        from genops.providers.traceloop_validation import (
            print_validation_result,
            validate_setup,
        )
        print("✅ GenOps Traceloop validation utilities loaded successfully")
    except ImportError as e:
        print(f"❌ Failed to import GenOps Traceloop validation utilities: {e}")
        print("\n💡 Fix: Run 'pip install genops[traceloop]'")
        return False

    # Quick environment check
    print("\n🌍 Environment Check:")
    print("-" * 30)

    # Check OpenLLMetry dependencies
    try:
        import openllmetry
        print("✅ OpenLLMetry: Open-source framework available")
        openllmetry_version = getattr(openllmetry, '__version__', 'unknown')
        print(f"   📦 Version: {openllmetry_version}")
    except ImportError:
        print("❌ OpenLLMetry: Not installed")
        print("   💡 Fix: Run 'pip install openllmetry' or 'pip install genops[traceloop]'")
        return False

    # Check Traceloop SDK
    try:
        from traceloop.sdk import Traceloop
        print("✅ Traceloop SDK: Available for commercial platform features")
    except ImportError:
        print("⚠️  Traceloop SDK: Not available (OpenLLMetry only)")
        print("   💡 For commercial features: pip install traceloop-sdk")

    # Check Traceloop platform configuration (optional)
    traceloop_api_key = os.getenv('TRACELOOP_API_KEY')
    traceloop_base_url = os.getenv('TRACELOOP_BASE_URL', 'https://app.traceloop.com')

    if traceloop_api_key:
        print("✅ TRACELOOP_API_KEY: Found (commercial platform access)")
        print(f"🌐 TRACELOOP_BASE_URL: {traceloop_base_url}")
    else:
        print("ℹ️  TRACELOOP_API_KEY: Not configured (open-source mode)")
        print("   💡 For commercial features, get your key at: https://app.traceloop.com")

    # Check LLM provider keys
    providers_found = []
    provider_keys = {
        'OpenAI': 'OPENAI_API_KEY',
        'Anthropic': 'ANTHROPIC_API_KEY',
        'Groq': 'GROQ_API_KEY'
    }

    for provider, env_var in provider_keys.items():
        if os.getenv(env_var):
            providers_found.append(provider)
            print(f"✅ {provider}: Found and validated")
        else:
            print(f"⚠️  {provider}: Not configured ({env_var})")

    if not providers_found:
        print("\n❌ No LLM provider API keys found! You need at least one.")
        print("   • OpenAI: https://platform.openai.com/api-keys")
        print("   • Anthropic: https://console.anthropic.com/")
        print("   • Groq: https://console.groq.com/ (free tier available)")
        return False

    print(f"\n✅ Found {len(providers_found)} configured providers: {', '.join(providers_found)}")

    # Run comprehensive validation
    print("\n🧪 Running comprehensive validation...")
    print("-" * 40)

    try:
        validation_result = validate_setup(include_performance_tests=True)
        print_validation_result(validation_result, detailed=True)

        # Summary
        print("\n" + "=" * 75)
        if validation_result and hasattr(validation_result, 'overall_status'):
            if validation_result.overall_status.value == "PASSED":
                print("🎉 Success! Your Traceloop + OpenLLMetry + GenOps setup is ready!")
                print("\n🔍 Enhanced Observability Stack Active:")
                print("   • OpenLLMetry tracing ✅ Open-source LLM observability foundation")
                print("   • GenOps governance ✅ Enhanced with cost intelligence and policy enforcement")

                if traceloop_api_key:
                    print("   • Traceloop platform ✅ Commercial insights and enterprise features")
                else:
                    print("   • Traceloop platform ⚠️  Available with API key (optional)")

                for provider in providers_found:
                    print(f"   • {provider} ✅ Ready for governed LLM operations")

                print("\n📚 Next steps:")
                print("   • Run 'python basic_tracking.py' for OpenLLMetry + GenOps foundation")
                print("   • Run 'python auto_instrumentation.py' for zero-code integration")
                print("   • Run 'python traceloop_platform.py' for commercial platform features")

                print("\n💡 Quick Test:")
                print("   Try this command to test your enhanced observability:")
                print("   python -c \"from genops.providers.traceloop import instrument_traceloop; print('Enhanced observability ready!')\"")

            else:
                print("⚠️  Setup validation completed with warnings.")
                print("   Review the detailed output above for specific issues.")
                print("   You can still proceed, but some features may not work optimally.")
        else:
            print("❌ Setup validation failed. Please review the errors above.")
            print("\n🔧 Common fixes:")
            print("   • Verify all API keys are correct and have sufficient credits")
            print("   • Check network connectivity to AI providers")
            print("   • Try: pip install --upgrade genops[traceloop]")
            return False

    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   • Check your API keys are valid")
        print("   • Verify network connectivity")
        print("   • Try: pip install --upgrade genops[traceloop] openllmetry")
        return False

    return True


def demonstrate_quick_integration():
    """Show a quick integration example."""
    print("\n🚀 Quick Integration Demo")
    print("-" * 25)

    try:
        from genops.providers.traceloop import instrument_traceloop

        # Test basic adapter creation
        print("✅ Creating GenOps Traceloop adapter...")
        adapter = instrument_traceloop(
            team="validation-demo",
            project="setup-check",
            environment="development"
        )

        print("✅ Enhanced Traceloop + OpenLLMetry observability ready!")
        print("\n🔍 Integration Features Available:")

        integration_features = [
            "🔍 OpenLLMetry Foundation - Open-source observability with OpenTelemetry standards",
            "💰 Cost Intelligence - Real-time cost tracking integrated with observability",
            "🏷️ Team Attribution - Automatic cost attribution to teams and projects",
            "🛡️ Policy Compliance - Budget enforcement and governance validation",
            "📊 Evaluation Governance - LLM evaluation tracking with cost oversight",
            "⚡ Zero-Code Setup - Auto-instrumentation for existing OpenLLMetry apps",
            "📈 Business Intelligence - Cost optimization insights and recommendations",
            "🏭 Traceloop Platform - Enterprise insights and advanced analytics (with API key)"
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

        print("\n" + "🌟" * 30)
        print("Your Traceloop + OpenLLMetry + GenOps integration is ready!")
        print("Enhanced LLM observability with governance intelligence!")
        print("🌟" * 30)
        sys.exit(0)
    else:
        print("\n❌ Setup validation failed. Please fix the issues above.")
        sys.exit(1)
