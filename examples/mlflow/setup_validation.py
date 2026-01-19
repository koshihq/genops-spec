"""
MLflow Setup Validation Example

This script validates your MLflow + GenOps setup and provides detailed
diagnostics with actionable fix suggestions.

Run this script to ensure everything is configured correctly before using
MLflow with GenOps governance tracking.

Usage:
    python examples/mlflow/setup_validation.py

Expected output:
    - Dependency checks (mlflow, opentelemetry, genops)
    - Configuration validation (tracking URI, governance attributes)
    - Connectivity tests (tracking server, model registry)
    - Governance feature validation
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Run MLflow setup validation."""
    print("\n" + "=" * 70)
    print("MLFLOW + GENOPS SETUP VALIDATION")
    print("=" * 70)
    print("\nValidating your MLflow setup for GenOps governance...")

    try:
        # Import validation functions
        from genops.providers.mlflow import validate_setup, print_validation_result

        # Run comprehensive validation
        result = validate_setup(
            check_connectivity=True,
            check_governance=True
        )

        # Print formatted results
        print_validation_result(result)

        # Return appropriate exit code
        if result.passed:
            print("\n✅ Validation PASSED")
            print("You're ready to use MLflow with GenOps governance!\n")
            return 0
        else:
            print("\n❌ Validation FAILED")
            print("Please fix the errors above before proceeding.\n")
            return 1

    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\nPossible fixes:")
        print("  1. Install required packages:")
        print("     pip install mlflow opentelemetry-api opentelemetry-sdk")
        print("  2. Install GenOps from source:")
        print("     pip install -e .")
        print()
        return 1

    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        print("\nPlease check your installation and configuration.\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
