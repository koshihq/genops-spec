"""
Example: Setup Validation for GenOps LangChain Integration
Demonstrates: How to verify your GenOps LangChain setup is working correctly
Use case: Troubleshooting and verifying installation before development
"""

import sys

# GenOps validation imports
try:
    from genops.providers.langchain import print_validation_result, validate_setup
except ImportError:
    print("❌ GenOps not installed. Run: pip install genops-ai[langchain]")
    sys.exit(1)


def main():
    """Run comprehensive setup validation."""
    print("🔍 GenOps LangChain Setup Validation")
    print("=" * 60)

    print("This utility will check your GenOps LangChain integration setup")
    print("and identify any issues that need to be resolved.\n")

    # Run validation
    print("Running validation checks...")
    result = validate_setup()

    # Print results
    print_validation_result(result)

    # Additional guidance based on results
    if result.is_valid:
        print("\n🎉 Your setup is ready to go!")
        print("\nNext steps:")
        print("   - Try basic_chain_tracking.py for a simple example")
        print("   - Explore auto_instrumentation.py for zero-code setup")
        print("   - Check multi_provider_costs.py for cost tracking")
    else:
        print("\n🔧 Setup needs attention before proceeding.")
        print("\nRecommended actions:")

        # Check for common issues and provide specific guidance
        errors = [issue for issue in result.issues if issue.level == "error"]

        has_env_errors = any(issue.component == "environment" for issue in errors)
        has_dep_errors = any(issue.component == "dependencies" for issue in errors)
        has_genops_errors = any(issue.component == "genops" for issue in errors)

        if has_env_errors:
            print("   1. 🔑 Set up your environment variables (API keys)")

        if has_dep_errors:
            print("   2. 📦 Install missing dependencies")

        if has_genops_errors:
            print("   3. 🔧 Fix GenOps installation")

        print("   4. 🔄 Run this validation script again")
        print("   5. 📚 Check the troubleshooting guide: docs/integrations/langchain.md")

    print(f"\n📊 Exit code: {'0' if result.is_valid else '1'}")
    return 0 if result.is_valid else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
