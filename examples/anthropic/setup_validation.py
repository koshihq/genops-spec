#!/usr/bin/env python3
"""
Anthropic Setup Validation Example

This script validates your Anthropic + GenOps setup and provides detailed diagnostics
for any configuration issues. Run this first before other examples.

Usage:
    python setup_validation.py

Prerequisites:
    pip install genops-ai[anthropic]
    export ANTHROPIC_API_KEY="your_anthropic_key_here"
"""

import os
import sys


def main():
    """Run comprehensive Anthropic + GenOps setup validation."""
    print("🔍 Anthropic + GenOps Setup Validation")
    print("=" * 50)

    # Import validation utilities
    try:
        from genops.providers.anthropic_validation import (
            print_validation_result,
            validate_setup,
        )
        print("✅ GenOps Anthropic validation utilities loaded successfully")
    except ImportError as e:
        print(f"❌ Failed to import GenOps Anthropic validation utilities: {e}")
        print("\n💡 Fix: Run 'pip install genops-ai[anthropic]'")
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
            print("🎉 Success! Your Anthropic + GenOps setup is ready to use.")
            print("\n📚 Next steps:")
            print("   • Run 'python basic_tracking.py' for simple tracking")
            print("   • Run 'python auto_instrumentation.py' for zero-code setup")
            print("   • Check out cost_optimization.py for Claude model selection")
            return True
        else:
            print("⚠️  Setup validation found issues that need attention.")
            print("\n💡 Please fix the errors above and run validation again.")
            return False

    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        print("\n🐛 Debug information:")
        print(f"   • Python version: {sys.version}")
        print(f"   • Anthropic API key set: {bool(os.getenv('ANTHROPIC_API_KEY'))}")
        print(f"   • Current working directory: {os.getcwd()}")
        return False

def manual_check():
    """Perform manual validation checks as fallback."""
    print("\n🔧 Manual Validation Checks")
    print("-" * 30)

    issues = []

    # Check Anthropic API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY environment variable not set")
        issues.append("Set ANTHROPIC_API_KEY environment variable")
    elif not api_key.startswith("sk-ant-"):
        print("⚠️  ANTHROPIC_API_KEY doesn't look like a valid Anthropic key (should start with 'sk-ant-')")
        issues.append("Verify ANTHROPIC_API_KEY format")
    else:
        # Security: Never log API key content, even partially
        print("✅ ANTHROPIC_API_KEY is set and properly formatted")

    # Check GenOps installation
    try:
        import genops
        print(f"✅ GenOps package imported successfully (version: {getattr(genops, '__version__', 'unknown')})")
    except ImportError as e:
        print(f"❌ Failed to import genops: {e}")
        issues.append("Install genops with: pip install genops-ai[anthropic]")

    # Check Anthropic installation
    try:
        import anthropic
        print(f"✅ Anthropic package imported successfully (version: {getattr(anthropic, '__version__', 'unknown')})")
    except ImportError as e:
        print(f"❌ Failed to import anthropic: {e}")
        issues.append("Install anthropic with: pip install anthropic")

    # Check OpenTelemetry (optional)
    try:
        import opentelemetry
        _ = opentelemetry.__name__  # Reference to avoid unused import warning
        print("✅ OpenTelemetry is available")

        # Check if OTLP endpoint is configured
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if otlp_endpoint:
            print(f"✅ OTLP endpoint configured: {otlp_endpoint}")
        else:
            print("ℹ️  No OTLP endpoint configured (optional for basic usage)")

    except ImportError:
        print("⚠️  OpenTelemetry not available (optional)")

    # Test basic Anthropic connectivity (if key is available)
    if api_key and api_key.startswith("sk-ant-"):
        try:
            from anthropic import Anthropic
            client = Anthropic()

            # Simple test call
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )

            if response and hasattr(response, 'content') and response.content:
                print("✅ Anthropic API connectivity test successful")
            else:
                print("⚠️  Anthropic API returned unexpected response format")
                issues.append("Check Anthropic API response handling")

        except Exception as e:
            print(f"❌ Anthropic API connectivity test failed: {e}")
            issues.append("Verify Anthropic API key and network connectivity")

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
        print("\n✨ Ready to explore Anthropic + GenOps examples!")
        sys.exit(0)
    else:
        print("\n❌ Setup validation failed. Please fix the issues above.")
        sys.exit(1)
