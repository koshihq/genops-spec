#!/usr/bin/env python3
"""
Arize AI + GenOps Setup Validation Example

This example demonstrates comprehensive validation of Arize AI integration
with GenOps governance. It checks SDK installation, authentication, 
configuration, and provides actionable troubleshooting guidance.

Features demonstrated:
- Complete setup validation with detailed diagnostics
- Actionable error messages with specific fix suggestions
- Environment variable validation and guidance
- Authentication testing with live connectivity checks
- Governance configuration validation

Run this example:
    python setup_validation.py

Expected output:
    ✅ All validation checks passed
    🚀 Ready to use Arize AI with GenOps governance
"""

import os
import sys
from typing import Any, Dict


def print_header():
    """Print example header information."""
    print("=" * 60)
    print("🔍 Arize AI + GenOps Setup Validation")
    print("=" * 60)
    print()


def check_environment_setup() -> Dict[str, Any]:
    """Check and display current environment configuration."""
    print("📋 Environment Configuration Check:")

    # Required environment variables
    required_vars = [
        "ARIZE_API_KEY",
        "ARIZE_SPACE_KEY"
    ]

    # Recommended environment variables
    recommended_vars = [
        "GENOPS_TEAM",
        "GENOPS_PROJECT",
        "GENOPS_ENVIRONMENT",
        "GENOPS_DAILY_BUDGET_LIMIT"
    ]

    env_status = {
        "required_missing": [],
        "recommended_missing": [],
        "configured": {}
    }

    # Check required variables
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values for display
            if "KEY" in var:
                masked_value = f"{value[:8]}..." if len(value) > 8 else "***"
                env_status["configured"][var] = masked_value
                print(f"  ✅ {var}: {masked_value}")
            else:
                env_status["configured"][var] = value
                print(f"  ✅ {var}: {value}")
        else:
            env_status["required_missing"].append(var)
            print(f"  ❌ {var}: Not set (required)")

    # Check recommended variables
    for var in recommended_vars:
        value = os.getenv(var)
        if value:
            env_status["configured"][var] = value
            print(f"  ✅ {var}: {value}")
        else:
            env_status["recommended_missing"].append(var)
            print(f"  ⚠️  {var}: Not set (recommended)")

    print()
    return env_status


def run_comprehensive_validation():
    """Run comprehensive GenOps Arize validation."""
    print("🔍 Running Comprehensive Validation...")
    print()

    try:
        from genops.providers.arize_validation import (
            print_validation_result,
            validate_setup,
        )

        # Run complete validation
        result = validate_setup()

        # Print detailed results
        print_validation_result(result)

        return result

    except ImportError as e:
        print(f"❌ GenOps Arize provider not available: {e}")
        print("   Fix: pip install genops[arize]")
        return None
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return None


def run_quick_validation_checks():
    """Run quick validation checks for immediate feedback."""
    print("⚡ Quick Validation Checks:")

    checks_passed = 0
    total_checks = 4

    # Check 1: GenOps installation
    try:
        import genops
        print("  ✅ GenOps package installed")
        checks_passed += 1
    except ImportError:
        print("  ❌ GenOps package not installed")
        print("     Fix: pip install genops")

    # Check 2: Arize SDK availability
    try:
        import arize
        print(f"  ✅ Arize AI SDK installed (version: {getattr(arize, '__version__', 'unknown')})")
        checks_passed += 1
    except ImportError:
        print("  ❌ Arize AI SDK not installed")
        print("     Fix: pip install arize>=6.0.0")

    # Check 3: GenOps Arize provider
    try:
        from genops.providers.arize import GenOpsArizeAdapter
        print("  ✅ GenOps Arize provider available")
        checks_passed += 1
    except ImportError:
        print("  ❌ GenOps Arize provider not available")
        print("     Fix: pip install genops[arize]")

    # Check 4: Basic credentials
    api_key = os.getenv('ARIZE_API_KEY')
    space_key = os.getenv('ARIZE_SPACE_KEY')

    if api_key and space_key and len(api_key) > 10 and len(space_key) > 10:
        print("  ✅ Arize credentials configured")
        checks_passed += 1
    else:
        print("  ❌ Arize credentials missing or invalid")
        print("     Fix: Set ARIZE_API_KEY and ARIZE_SPACE_KEY environment variables")

    print(f"\n📊 Quick Check Results: {checks_passed}/{total_checks} passed")
    print()

    return checks_passed == total_checks


def demonstrate_adapter_creation():
    """Demonstrate creating a GenOps Arize adapter with validation."""
    print("🔧 Adapter Creation Test:")

    try:
        from genops.providers.arize import GenOpsArizeAdapter

        # Create adapter with environment-based configuration
        adapter = GenOpsArizeAdapter(
            team=os.getenv('GENOPS_TEAM', 'example-team'),
            project=os.getenv('GENOPS_PROJECT', 'setup-validation'),
            environment=os.getenv('GENOPS_ENVIRONMENT', 'development'),
            daily_budget_limit=float(os.getenv('GENOPS_DAILY_BUDGET_LIMIT', '20.0')),
            enable_cost_alerts=True,
            enable_governance=True
        )

        print("  ✅ GenOps Arize adapter created successfully")

        # Display adapter configuration
        print("  📋 Configuration:")
        print(f"     • Team: {adapter.team}")
        print(f"     • Project: {adapter.project}")
        print(f"     • Environment: {adapter.environment}")
        print(f"     • Daily Budget Limit: ${adapter.daily_budget_limit:.2f}")
        print(f"     • Cost Alerts: {adapter.enable_cost_alerts}")
        print(f"     • Governance: {adapter.enable_governance}")

        # Get adapter metrics
        metrics = adapter.get_metrics()
        print("  📊 Current Metrics:")
        print(f"     • Daily Usage: ${metrics['daily_usage']:.2f}")
        print(f"     • Budget Remaining: ${metrics['budget_remaining']:.2f}")
        print(f"     • Operations Count: {metrics['operation_count']}")
        print(f"     • Active Sessions: {metrics['active_monitoring_sessions']}")

        return True

    except Exception as e:
        print(f"  ❌ Adapter creation failed: {e}")
        return False


def test_auto_instrumentation():
    """Test auto-instrumentation functionality."""
    print("🤖 Auto-Instrumentation Test:")

    try:
        from genops.providers.arize import auto_instrument, get_current_adapter

        # Test auto-instrumentation setup
        adapter = auto_instrument(
            team=os.getenv('GENOPS_TEAM', 'example-team'),
            project=os.getenv('GENOPS_PROJECT', 'setup-validation'),
            enable_cost_alerts=False,  # Disable alerts for testing
            daily_budget_limit=100.0
        )

        print("  ✅ Auto-instrumentation enabled successfully")

        # Verify global adapter is set
        current_adapter = get_current_adapter()
        if current_adapter:
            print("  ✅ Global adapter configured")
        else:
            print("  ⚠️  Global adapter not set (this may be expected)")

        return True

    except Exception as e:
        print(f"  ❌ Auto-instrumentation test failed: {e}")
        return False


def provide_next_steps(validation_passed: bool):
    """Provide next steps based on validation results."""
    print("🚀 Next Steps:")

    if validation_passed:
        print("  ✅ Setup validation completed successfully!")
        print("  🎉 You're ready to use Arize AI with GenOps governance!")
        print()
        print("  📖 Try these examples next:")
        print("     • python basic_tracking.py       # Basic monitoring with governance")
        print("     • python auto_instrumentation.py # Zero-code integration")
        print("     • python cost_optimization.py    # Cost intelligence features")
        print()
        print("  📚 Additional resources:")
        print("     • Integration guide: docs/integrations/arize.md")
        print("     • All examples: examples/arize/README.md")
        print("     • GitHub issues: https://github.com/KoshiHQ/GenOps-AI/issues")
    else:
        print("  ❌ Setup validation found issues that need to be addressed")
        print("  🔧 Common fixes:")
        print("     • Install dependencies: pip install genops[arize]")
        print("     • Set environment variables:")
        print("       export ARIZE_API_KEY='your-api-key'")
        print("       export ARIZE_SPACE_KEY='your-space-key'")
        print("       export GENOPS_TEAM='your-team'")
        print("       export GENOPS_PROJECT='your-project'")
        print()
        print("  📋 Re-run validation after fixes:")
        print("     python setup_validation.py")


def print_system_info():
    """Print system information for debugging."""
    print("💻 System Information:")
    print(f"  • Python Version: {sys.version.split()[0]}")
    print(f"  • Platform: {sys.platform}")

    # Check installed packages
    try:
        import genops
        print(f"  • GenOps Version: {getattr(genops, '__version__', 'unknown')}")
    except ImportError:
        print("  • GenOps: Not installed")

    try:
        import arize
        print(f"  • Arize Version: {getattr(arize, '__version__', 'unknown')}")
    except ImportError:
        print("  • Arize: Not installed")

    try:
        import pandas
        print(f"  • Pandas Version: {pandas.__version__}")
    except ImportError:
        print("  • Pandas: Not installed")

    print()


def main():
    """Main validation workflow."""
    print_header()
    print_system_info()

    # Step 1: Check environment setup
    env_status = check_environment_setup()

    # Step 2: Run quick validation checks
    quick_checks_passed = run_quick_validation_checks()

    # Step 3: Test adapter creation
    adapter_creation_success = False
    if quick_checks_passed:
        adapter_creation_success = demonstrate_adapter_creation()
        print()

    # Step 4: Test auto-instrumentation
    auto_instrumentation_success = False
    if adapter_creation_success:
        auto_instrumentation_success = test_auto_instrumentation()
        print()

    # Step 5: Run comprehensive validation
    validation_result = None
    if quick_checks_passed:
        validation_result = run_comprehensive_validation()

    # Determine overall success
    overall_success = (
        quick_checks_passed and
        adapter_creation_success and
        validation_result is not None and
        validation_result.is_valid
    )

    # Step 6: Provide next steps
    provide_next_steps(overall_success)

    # Return appropriate exit code
    return 0 if overall_success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
