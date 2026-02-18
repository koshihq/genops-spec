#!/usr/bin/env python3
"""
LiteLLM + GenOps Setup Validation

Comprehensive validation script for LiteLLM integration with GenOps.
This script checks all requirements and provides actionable feedback
for setting up the most high-leverage GenOps integration.

Usage:
    python setup_validation.py          # Full validation
    python setup_validation.py --quick  # Essential checks only
    python setup_validation.py --test   # Include connectivity tests

Features:
    - LiteLLM installation and version checking
    - Provider API key validation across 100+ providers
    - GenOps integration functionality testing
    - Environment configuration verification
    - Actionable fix suggestions for all issues
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from genops.providers.litellm_validation import (
        print_validation_result,
        validate_litellm_setup,
    )
except ImportError as e:
    print("❌ Error: Cannot import GenOps LiteLLM validation module")
    print(f"   {e}")
    print("\n💡 Fix: Install GenOps with LiteLLM support:")
    print("   pip install genops[litellm]")
    sys.exit(1)


def main():
    """Run LiteLLM + GenOps validation with command line options."""

    parser = argparse.ArgumentParser(
        description="Validate LiteLLM + GenOps integration setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python setup_validation.py                    # Full validation
    python setup_validation.py --quick            # Quick essential checks
    python setup_validation.py --test             # Include API connectivity tests
    python setup_validation.py --quiet            # Minimal output

This validation covers the highest-leverage GenOps integration:
• Single integration point for 100+ LLM providers
• Unified cost tracking and governance across entire ecosystem
• Provider-agnostic budget controls and compliance monitoring
        """,
    )

    parser.add_argument(
        "--quick", action="store_true", help="Run only essential validations (faster)"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Include API connectivity tests (requires API keys)",
    )

    parser.add_argument(
        "--quiet", action="store_true", help="Minimal output - show only summary"
    )

    parser.add_argument(
        "--providers",
        nargs="*",
        help="Test specific providers only (e.g. --providers openai anthropic)",
    )

    args = parser.parse_args()

    # Print header unless quiet
    if not args.quiet:
        print("🚀 LiteLLM + GenOps Integration Validation")
        print("═" * 50)
        print("Testing the highest-leverage GenOps integration:")
        print("• Single instrumentation → 100+ LLM providers")
        print("• Unified governance across entire ecosystem")
        print("• Provider-agnostic cost tracking & compliance")

        if args.quick:
            print("\n🏃‍♂️ Running quick validation...")
        elif args.test:
            print("\n🔍 Running comprehensive validation with connectivity tests...")
        else:
            print("\n🔍 Running comprehensive validation...")

    # Run validation
    try:
        result = validate_litellm_setup(quick=args.quick, test_connectivity=args.test)

        # Print results
        print_validation_result(result, verbose=not args.quiet)

        # Additional guidance based on results
        if not args.quiet:
            if result.is_valid:
                print("\n🎯 Next Steps:")
                print("1. Try the basic auto-instrumentation example:")
                print("   python auto_instrumentation.py")
                print("\n2. Explore multi-provider cost tracking:")
                print("   python multi_provider_costs.py")
                print("\n3. See production patterns:")
                print("   python production_patterns.py")

            else:
                print("\n🔧 Recommended Actions:")
                print("1. Fix the critical errors shown above")
                print("2. Re-run validation: python setup_validation.py")
                print("3. Check documentation: https://docs.litellm.ai/")

        # Exit with appropriate code
        return 0 if result.is_valid else 1

    except KeyboardInterrupt:
        print("\n\n⚠️ Validation interrupted by user")
        return 1

    except Exception as e:
        print(f"\n❌ Validation failed with unexpected error: {e}")
        if not args.quiet:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
