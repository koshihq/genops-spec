#!/usr/bin/env python3
"""
CrewAI + GenOps Setup Validation

Comprehensive validation and troubleshooting for CrewAI integration with GenOps.
Run this first to ensure your environment is properly configured.

Usage:
    python setup_validation.py [--quick]

Options:
    --quick     Skip comprehensive tests (faster validation)

Features:
    - CrewAI framework detection and version validation
    - AI provider configuration verification
    - Environment variable and API key validation
    - GenOps component compatibility checks
    - Integration testing with sample crew execution
    - Actionable error messages with fix suggestions
"""

import argparse
import sys


def main():
    """Run comprehensive setup validation."""
    parser = argparse.ArgumentParser(description="Validate CrewAI + GenOps setup")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip comprehensive tests for faster validation",
    )
    args = parser.parse_args()

    # Try to import GenOps validation
    try:
        from genops.providers.crewai import (
            print_validation_result,
            validate_crewai_setup,
        )
    except ImportError as e:
        print("❌ GenOps CrewAI provider not available")
        print(f"   Error: {e}")
        print("\n🔧 Fix: Install GenOps with CrewAI support:")
        print("   pip install genops-ai[crewai]")
        return 1

    print("🔍 CrewAI + GenOps Setup Validation")
    print("=" * 40)

    if args.quick:
        print("⚡ Running quick validation (use --comprehensive for full tests)")
    else:
        print("🔬 Running comprehensive validation...")

    # Run validation
    result = validate_crewai_setup(quick=args.quick)

    # Print results
    print_validation_result(result)

    # Return appropriate exit code
    return 0 if result.is_valid else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Validation failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
