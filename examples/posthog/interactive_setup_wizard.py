#!/usr/bin/env python3
"""
PostHog + GenOps Interactive Setup Wizard

This script provides an interactive command-line wizard to help users configure
their PostHog + GenOps integration step by step. It guides through API key setup,
team configuration, budget limits, and validates the complete setup.

Usage:
    python interactive_setup_wizard.py

Features:
- Step-by-step environment configuration
- API key validation and format checking
- Team and project attribution setup
- Budget limit configuration with cost guidance
- Automatic validation after configuration
- Clear next steps and documentation links

Prerequisites:
    pip install genops[posthog]

Author: GenOps AI Team
License: Apache 2.0
"""

import sys


def main():
    """Run the interactive PostHog + GenOps setup wizard."""
    print("🧙 PostHog + GenOps Interactive Setup Wizard")
    print("=" * 50)
    print()
    print("This wizard will guide you through setting up PostHog with GenOps governance.")
    print("It's perfect for first-time users or when setting up new environments.")
    print()

    try:
        # Import the interactive setup wizard
        from genops.providers.posthog_validation import interactive_setup_wizard

        print("✅ GenOps PostHog integration available")
        print()

        # Run the interactive wizard
        interactive_setup_wizard()

    except ImportError as e:
        print(f"❌ Failed to import GenOps PostHog validation: {e}")
        print()
        print("🔧 Fix: Install GenOps with PostHog support:")
        print("   pip install genops[posthog]")
        print()
        print("📚 Documentation: https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/posthog")
        return False

    except Exception as e:
        print(f"💥 Unexpected error during setup wizard: {e}")
        print()
        print("🐛 Please report this issue: https://github.com/KoshiHQ/GenOps-AI/issues")
        print("📧 Or try manual setup: python examples/posthog/setup_validation.py")
        return False

    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Setup wizard interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error in setup wizard: {e}")
        sys.exit(1)
