#!/usr/bin/env python3
"""
OpenAI Setup Validation Example

This script validates your OpenAI + GenOps setup and provides detailed diagnostics
for any configuration issues. Run this first before other examples.

Usage:
    python setup_validation.py

Prerequisites:
    pip install genops-ai[openai]
    export OPENAI_API_KEY="your_api_key_here"
"""

import os
import sys


def main():
    """Run comprehensive OpenAI + GenOps setup validation."""
    print("🔍 OpenAI + GenOps Setup Validation")
    print("=" * 50)

    # Import validation utilities
    try:
        from genops.providers.openai_validation import (
            print_validation_result,
            validate_setup,
        )
        print("✅ GenOps OpenAI validation utilities loaded successfully")
    except ImportError as e:
        print(f"❌ Failed to import GenOps OpenAI validation utilities: {e}")
        print("\n💡 Fix: Run 'pip install genops-ai[openai]'")
        return False

    # Run comprehensive validation
    print("\n🧪 Running validation checks...")
    print("-" * 30)

    try:
        validation_result = validate_setup()
        print_validation_result(validation_result)

        # Summary
        print("\n" + "=" * 50)
        if validation_result and validation_result.is_valid:
            print("🎉 Success! Your OpenAI + GenOps setup is ready to use.")
            print("\n📚 Next steps:")
            print("   • Run 'python basic_tracking.py' for simple tracking")
            print("   • Run 'python auto_instrumentation.py' for zero-code setup")
            print("   • Check out cost_optimization.py for advanced patterns")
            return True
        else:
            print("⚠️  Setup validation found issues that need attention.")
            print("\n💡 Please fix the errors above and run validation again.")
            return False

    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        print("\n🐛 Debug information:")
        print(f"   • Python version: {sys.version}")
        print(f"   • OpenAI API key set: {bool(os.getenv('OPENAI_API_KEY'))}")
        print(f"   • Current working directory: {os.getcwd()}")
        return False

def manual_check():
    """Perform manual validation checks as fallback."""
    print("\n🔧 Manual Validation Checks")
    print("-" * 30)

    issues = []

    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        issues.append("Set OPENAI_API_KEY environment variable")
    elif not api_key.startswith("sk-"):
        print("⚠️  OPENAI_API_KEY doesn't look like a valid OpenAI key (should start with 'sk-')")
        issues.append("Verify OPENAI_API_KEY format")
    else:
        # Security: Never log API key content, even partially
        print("✅ OPENAI_API_KEY is set and properly formatted")

    # Check GenOps installation
    try:
        import genops
        print(f"✅ GenOps package imported successfully (version: {getattr(genops, '__version__', 'unknown')})")
    except ImportError as e:
        print(f"❌ Failed to import genops: {e}")
        issues.append("Install genops with: pip install genops-ai[openai]")

    # Check OpenAI installation
    try:
        import openai
        print(f"✅ OpenAI package imported successfully (version: {getattr(openai, '__version__', 'unknown')})")
    except ImportError as e:
        print(f"❌ Failed to import openai: {e}")
        issues.append("Install openai with: pip install openai")

    # Check OpenTelemetry (optional)
    try:
        import opentelemetry
        opentelemetry.__name__  # Reference to avoid unused import warning
        print("✅ OpenTelemetry is available")

        # Check if OTLP endpoint is configured
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if otlp_endpoint:
            print(f"✅ OTLP endpoint configured: {otlp_endpoint}")
        else:
            print("ℹ️  No OTLP endpoint configured (optional for basic usage)")

    except ImportError:
        print("⚠️  OpenTelemetry not available (optional)")

    # Test basic OpenAI connectivity (if key is available)
    if api_key and api_key.startswith("sk-"):
        try:
            from openai import OpenAI
            client = OpenAI()

            # Simple test call
            models = client.models.list()
            if models:
                print("✅ OpenAI API connectivity test successful")
            else:
                print("⚠️  OpenAI API returned empty models list")
                issues.append("Check OpenAI API key permissions")

        except Exception as e:
            print(f"❌ OpenAI API connectivity test failed: {e}")
            issues.append("Verify OpenAI API key and network connectivity")

    # Summary
    print("\n" + "=" * 50)
    if not issues:
        print("🎉 Manual validation passed! Setup appears to be correct.")
        return True
    else:
        print(f"⚠️  Found {len(issues)} issues:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        return False

if __name__ == "__main__":
    success = main()

    if not success:
        print("\n" + "=" * 50)
        print("🔧 Falling back to manual validation...")
        success = manual_check()

    if success:
        print("\n✨ Ready to explore OpenAI + GenOps examples!")
        sys.exit(0)
    else:
        print("\n❌ Setup validation failed. Please fix the issues above.")
        sys.exit(1)
