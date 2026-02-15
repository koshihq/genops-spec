#!/usr/bin/env python3
"""
Vercel AI SDK Setup Validation Script

This script validates that your environment is properly configured
for GenOps integration with Vercel AI SDK.

Usage:
    python setup_validation.py
    python setup_validation.py --quick
    python setup_validation.py --full
"""

import argparse
import sys


def main():
    """Main validation script."""
    parser = argparse.ArgumentParser(
        description="Validate Vercel AI SDK integration with GenOps"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick validation (basic checks only)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full validation including provider connectivity",
    )
    parser.add_argument(
        "--no-nodejs", action="store_true", help="Skip Node.js validation"
    )
    parser.add_argument(
        "--no-npm", action="store_true", help="Skip npm package validation"
    )

    args = parser.parse_args()

    try:
        from genops.providers.vercel_ai_sdk_validation import validate_setup
    except ImportError as e:
        print(f"❌ GenOps not installed: {e}")
        print("Install with: pip install genops")
        sys.exit(1)

    # Determine validation scope
    if args.quick:
        check_nodejs = not args.no_nodejs
        check_npm_packages = False
        check_python_deps = True
        check_environment = False
        check_genops_config = True
        check_provider_access = False
    elif args.full:
        check_nodejs = not args.no_nodejs
        check_npm_packages = not args.no_npm
        check_python_deps = True
        check_environment = True
        check_genops_config = True
        check_provider_access = True
    else:
        # Default validation
        check_nodejs = not args.no_nodejs
        check_npm_packages = not args.no_npm
        check_python_deps = True
        check_environment = True
        check_genops_config = True
        check_provider_access = False

    # Run validation
    result = validate_setup(
        check_nodejs=check_nodejs,
        check_npm_packages=check_npm_packages,
        check_python_deps=check_python_deps,
        check_environment=check_environment,
        check_genops_config=check_genops_config,
        check_provider_access=check_provider_access,
        verbose=True,
    )

    # Exit with appropriate code
    sys.exit(0 if result.all_passed else 1)


if __name__ == "__main__":
    main()
