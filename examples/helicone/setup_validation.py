#!/usr/bin/env python3
"""
Helicone AI Gateway Setup Validation Example

This script validates your Helicone + GenOps setup across multiple AI providers
and provides detailed diagnostics for any configuration issues. Run this first 
before other examples.

Usage:
    python setup_validation.py

Prerequisites:
    pip install genops[helicone]
    export HELICONE_API_KEY="your_helicone_api_key"
    export OPENAI_API_KEY="your_openai_api_key"  # At least one provider required
"""

import os
import sys


def main():
    """Run comprehensive Helicone + GenOps setup validation."""
    print("🔍 Helicone AI Gateway + GenOps Setup Validation")
    print("=" * 60)

    # Import validation utilities
    try:
        from genops.providers.helicone_validation import (
            print_validation_result,
            validate_setup,
        )
        print("✅ GenOps Helicone validation utilities loaded successfully")
    except ImportError as e:
        print(f"❌ Failed to import GenOps Helicone validation utilities: {e}")
        print("\n💡 Fix: Run 'pip install genops[helicone]'")
        return False

    # Quick environment check
    print("\n🌍 Environment Check:")
    print("-" * 30)

    helicone_key = os.getenv('HELICONE_API_KEY')
    if helicone_key:
        print(f"✅ HELICONE_API_KEY: Found (ends with: ...{helicone_key[-6:]})")
    else:
        print("❌ HELICONE_API_KEY: Not found")
        print("   Get your key at: https://app.helicone.ai/")

    # Check provider keys
    providers_found = []
    provider_keys = {
        'OpenAI': 'OPENAI_API_KEY',
        'Anthropic': 'ANTHROPIC_API_KEY',
        'Groq': 'GROQ_API_KEY',
        'Vertex AI': 'VERTEX_AI_CREDENTIALS'
    }

    for provider, env_var in provider_keys.items():
        if os.getenv(env_var):
            providers_found.append(provider)
            key_val = os.getenv(env_var)
            if env_var == 'VERTEX_AI_CREDENTIALS':
                print(f"✅ {provider}: Found ({key_val})")
            else:
                print(f"✅ {provider}: Found (ends with: ...{key_val[-6:]})")
        else:
            print(f"⚠️  {provider}: Not configured ({env_var})")

    if not providers_found:
        print("\n❌ No provider API keys found! You need at least one.")
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
        print("\n" + "=" * 60)
        if validation_result and hasattr(validation_result, 'overall_status'):
            if validation_result.overall_status == "PASSED":
                print("🎉 Success! Your Helicone AI Gateway + GenOps setup is ready!")
                print("\n🚀 Multi-Provider Gateway Active:")
                for provider in providers_found:
                    print(f"   • {provider} ✅ Ready for intelligent routing")

                print("\n📚 Next steps:")
                print("   • Run 'python basic_tracking.py' for multi-provider tracking")
                print("   • Run 'python multi_provider_costs.py' for cost comparison")
                print("   • Run 'python cost_optimization.py' for intelligent routing")

                print("\n💡 Quick Test:")
                print("   Try this command to test your gateway:")
                print("   python -c \"from genops.providers.helicone import instrument_helicone; print('Gateway ready!')\"")

            else:
                print("⚠️  Setup validation completed with warnings.")
                print("   Review the detailed output above for specific issues.")
                print("   You can still proceed, but some features may not work optimally.")
        else:
            print("❌ Setup validation failed. Please review the errors above.")
            print("\n🔧 Common fixes:")
            print("   • Verify all API keys are correct and have sufficient credits")
            print("   • Check network connectivity to AI providers")
            print("   • Ensure Helicone gateway is accessible")
            return False

    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   • Check your API keys are valid")
        print("   • Verify network connectivity")
        print("   • Try: pip install --upgrade genops[helicone]")
        return False

    return True


if __name__ == "__main__":
    """Main entry point."""
    success = main()

    if success:
        print("\n" + "🌟" * 20)
        print("Your Helicone AI Gateway setup is ready!")
        print("Access 100+ AI models with unified cost tracking!")
        print("🌟" * 20)
        sys.exit(0)
    else:
        print("\n❌ Setup validation failed. Please fix the issues above.")
        sys.exit(1)
