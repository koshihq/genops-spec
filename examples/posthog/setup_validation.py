#!/usr/bin/env python3
"""
PostHog Product Analytics Setup Validation Example

This script validates your PostHog + GenOps setup for enhanced product analytics
with governance intelligence and provides detailed diagnostics for any configuration issues. 
Run this first before other examples.

Usage:
    python setup_validation.py

Prerequisites:
    pip install genops[posthog]
    export POSTHOG_API_KEY="phc_your-project-api-key"
    export GENOPS_TEAM="your-team-name"  # Optional but recommended
    export GENOPS_PROJECT="your-project-name"  # Optional but recommended
"""

import os
import sys


def main():
    """Run comprehensive PostHog + GenOps setup validation."""
    print("🔍 PostHog Product Analytics + GenOps Setup Validation")
    print("=" * 65)

    # Import validation utilities
    try:
        from genops.providers.posthog_validation import (
            print_validation_result,
            validate_setup,
        )
        print("✅ GenOps PostHog validation utilities loaded successfully")
    except ImportError as e:
        print(f"❌ Failed to import GenOps PostHog validation utilities: {e}")
        print("\n💡 Fix: Run 'pip install genops[posthog]'")
        return False

    # Quick environment check
    print("\n🌍 Environment Check:")
    print("-" * 30)

    api_key = os.getenv('POSTHOG_API_KEY')
    host = os.getenv('POSTHOG_HOST', 'https://app.posthog.com')
    team = os.getenv('GENOPS_TEAM')
    project = os.getenv('GENOPS_PROJECT')

    if api_key:
        if api_key.startswith('phc_'):
            print("✅ POSTHOG_API_KEY: Found and format validated")
        else:
            print("⚠️ POSTHOG_API_KEY: Found but format may be incorrect")
            print("   Expected format: phc_...")
    else:
        print("❌ POSTHOG_API_KEY: Not found")
        print("   Get your project API key at: https://app.posthog.com/project/settings")

    print(f"🌐 POSTHOG_HOST: {host}")

    if team:
        print(f"✅ GENOPS_TEAM: {team}")
    else:
        print("⚠️ GENOPS_TEAM: Not configured")
        print("   Set for better cost attribution")

    if project:
        print(f"✅ GENOPS_PROJECT: {project}")
    else:
        print("⚠️ GENOPS_PROJECT: Not configured")
        print("   Set for better cost attribution")

    # Check for commonly used analytics environments
    print("\n🔍 Analytics Environment Detection:")
    analytics_contexts = {
        'Jupyter Notebook': any(['jupyter' in str(sys.modules.get(mod, '')) for mod in sys.modules]),
        'Django': 'django' in sys.modules,
        'Flask': 'flask' in sys.modules,
        'FastAPI': 'fastapi' in sys.modules,
        'Streamlit': 'streamlit' in sys.modules
    }

    detected_contexts = [context for context, detected in analytics_contexts.items() if detected]
    if detected_contexts:
        print(f"📊 Detected analytics contexts: {', '.join(detected_contexts)}")
    else:
        print("📊 No specific analytics frameworks detected in current environment")

    print(f"\n{'='*65}")
    print("🔧 Running Comprehensive Validation...")
    print(f"{'='*65}")

    # Run comprehensive validation
    try:
        validation_result = validate_setup(verbose=True)
        print_validation_result(validation_result, show_successes=True)

        # Additional setup guidance
        print("\n" + "="*65)
        print("📚 Quick Setup Commands:")
        print("-" * 25)
        print("# Set up environment (replace with your values)")
        print("export POSTHOG_API_KEY='phc_your_project_api_key'")
        print("export GENOPS_TEAM='analytics-team'")
        print("export GENOPS_PROJECT='product-analytics'")
        print()
        print("# Install dependencies")
        print("pip install genops[posthog]")
        print()
        print("# Test basic functionality")
        print("python basic_tracking.py")

        if validation_result.is_valid:
            print("\n✅ Setup validation completed successfully!")
            print("🚀 You're ready to run the PostHog examples!")
            return True
        else:
            print(f"\n❌ Setup validation found {validation_result.error_count} issues")
            print("🔧 Please fix the issues above before proceeding")
            return False

    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure you have installed: pip install genops[posthog]")
        print("2. Check your PostHog API key configuration")
        print("3. Verify internet connectivity for PostHog API access")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Setup validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during validation: {e}")
        print("🐛 Please report this issue: https://github.com/KoshiHQ/GenOps-AI/issues")
        sys.exit(1)
