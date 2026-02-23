#!/usr/bin/env python3
"""
Databricks Unity Catalog Setup Validation

Validates your Databricks Unity Catalog setup for GenOps governance.
This script checks dependencies, configuration, connectivity, and governance features.

⭐ RUN THIS FIRST before trying any other examples!

Usage:
    python setup_validation.py
    python setup_validation.py --detailed --connectivity --governance
"""

import argparse
import logging
import sys
from pathlib import Path

# Try to import GenOps - handle both pip install and repo development
try:
    # First try normal pip install import
    from genops.providers.databricks_unity_catalog.validation import (
        print_validation_result,
        validate_setup,
    )

    _GENOPS_AVAILABLE = True
except ImportError:
    try:
        # Fallback to development repo structure
        sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
        from genops.providers.databricks_unity_catalog.validation import (
            print_validation_result,
            validate_setup,
        )

        _GENOPS_AVAILABLE = True
    except ImportError as e:
        print(f"❌ Error importing GenOps Databricks Unity Catalog provider: {e}")
        print("💡 Make sure you have installed genops[databricks]:")
        print("   pip install genops[databricks]")
        print("   Or run from the repository root directory")
        sys.exit(1)


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description="Validate Databricks Unity Catalog setup for GenOps governance"
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Enable detailed logging output"
    )
    parser.add_argument(
        "--connectivity",
        action="store_true",
        default=True,
        help="Test connectivity to Databricks (enabled by default)",
    )
    parser.add_argument(
        "--governance",
        action="store_true",
        default=True,
        help="Validate governance features (enabled by default)",
    )
    parser.add_argument("--workspace-url", help="Override Databricks workspace URL")

    args = parser.parse_args()

    # Configure logging
    if args.detailed:
        logging.basicConfig(level=logging.DEBUG)
        print("🔍 Debug logging enabled")
    else:
        logging.basicConfig(level=logging.WARNING)

    print("🚀 Starting Databricks Unity Catalog GenOps validation...")
    print(
        "   This will check dependencies, configuration, connectivity, and governance features."
    )
    print()

    # Run validation
    try:
        result = validate_setup(
            workspace_url=args.workspace_url,
            check_connectivity=args.connectivity,
            check_governance=args.governance,
        )

        # Print formatted result
        print_validation_result(result)

        # Exit with appropriate code
        if result.passed:
            print(
                "✨ Validation successful! You're ready to use Databricks Unity Catalog with GenOps."
            )
            print(
                "🎯 Next step: Try 'python basic_tracking.py' to see governance in action."
            )
            sys.exit(0)
        else:
            print("❌ Validation failed. Please fix the issues above and try again.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during validation: {e}")
        if args.detailed:
            import traceback

            traceback.print_exc()
        print("\n📧 If this error persists, please report it at:")
        print("   https://github.com/KoshiHQ/GenOps-AI/issues")
        sys.exit(1)


if __name__ == "__main__":
    main()
