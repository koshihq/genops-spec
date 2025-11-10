#!/usr/bin/env python3
"""
Hugging Face GenOps Setup Validation Example

This example demonstrates comprehensive validation of your Hugging Face GenOps setup.
Run this first to ensure everything is configured correctly.

Example usage:
    python setup_validation.py

Features demonstrated:
- Environment variable validation
- Dependency checking with fix suggestions  
- Hugging Face connectivity testing
- GenOps integration verification
- Cost calculation testing
"""

import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def main():
    """Run comprehensive Hugging Face GenOps setup validation."""

    print("🤗 Starting Hugging Face GenOps Setup Validation...")
    print("This will check your environment, dependencies, and integration setup.\n")

    try:
        # Import validation utilities
        from genops.providers.huggingface_validation import (
            print_huggingface_validation_result,
            validate_huggingface_setup,
        )

        # Run comprehensive validation
        result = validate_huggingface_setup()

        # Display results in user-friendly format
        print_huggingface_validation_result(result)

        # Exit with appropriate code
        if result.is_valid:
            print("✅ Validation passed! Your Hugging Face GenOps setup is ready to use.")
            return 0
        else:
            error_count = len([i for i in result.issues if i.level == "error"])
            print(f"❌ Validation found {error_count} error(s) that need to be fixed.")
            return 1

    except ImportError as e:
        print(f"❌ Could not import GenOps Hugging Face validation utilities: {e}")
        print("\n💡 Fix suggestions:")
        print("   1. Install GenOps AI: pip install genops-ai")
        print("   2. Install with Hugging Face support: pip install genops-ai[huggingface]")
        print("   3. Check your Python path and virtual environment")
        return 1

    except Exception as e:
        print(f"❌ Unexpected error during validation: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check your internet connection")
        print("   2. Verify Python environment and dependencies")
        print("   3. Report issue: https://github.com/KoshiHQ/GenOps-AI/issues")
        return 1


def quick_check():
    """Quick validation check for CI/automation use."""
    try:
        from genops.providers.huggingface_validation import quick_validate
        return quick_validate()
    except ImportError:
        print("❌ GenOps Hugging Face not available")
        return False
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    # Check for quick mode flag
    if len(sys.argv) > 1 and sys.argv[1] in ["--quick", "-q"]:
        success = quick_check()
        sys.exit(0 if success else 1)
    else:
        sys.exit(main())
