#!/usr/bin/env python3
"""
SkyRouter + GenOps Setup Validation Example

This example demonstrates how to validate your SkyRouter + GenOps configuration
before running production workloads. It provides comprehensive diagnostics for
multi-model routing setup, authentication, and governance configuration.

Features demonstrated:
- Environment variable validation for SkyRouter
- Multi-model routing configuration checks
- GenOps governance setup verification
- Interactive setup for missing configuration
- Actionable diagnostics with specific fix suggestions

Usage:
    export SKYROUTER_API_KEY="your-api-key"
    export GENOPS_TEAM="your-team"
    export GENOPS_PROJECT="your-project"
    python setup_validation.py

Author: GenOps AI Contributors
"""

import sys
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def run_validation():
    """Run comprehensive SkyRouter + GenOps setup validation."""

    print("🔍 SkyRouter + GenOps Setup Validation")
    print("=" * 50)
    print()

    try:
        from genops.providers.skyrouter_validation import (
            print_validation_result,
            validate_setup,
            validate_setup_interactive,
        )
    except ImportError as e:
        print(f"❌ Error importing GenOps SkyRouter validation: {e}")
        print(
            "💡 Make sure you're in the project root directory and GenOps is properly installed"
        )
        print("💡 Try: pip install genops[skyrouter]")
        return False

    # Step 1: Basic validation
    print("📋 Step 1: Running comprehensive validation checks...")
    print()

    result = validate_setup()
    print_validation_result(result, verbose=True)

    if result.is_valid:
        print()
        print("🎉 All validation checks passed!")
        print("🚀 Your SkyRouter integration is ready for multi-model routing")
        return True

    # Step 2: Interactive setup for issues
    print()
    print("🔧 Step 2: Let's fix the configuration issues...")
    print("Would you like to run interactive setup? (y/n): ", end="")

    try:
        user_input = input().strip().lower()
        if user_input in ["y", "yes", ""]:
            print()
            interactive_result = validate_setup_interactive()
            return interactive_result.is_valid
        else:
            print()
            print("👋 No problem! Fix the issues above and run validation again.")
            return False

    except KeyboardInterrupt:
        print()
        print("👋 Setup cancelled. Fix the issues above and run validation again.")
        return False


def demonstrate_configuration_examples():
    """Show configuration examples for different scenarios."""

    print("💡 Configuration Examples")
    print("=" * 30)
    print()

    print("🏗️ **Development Environment:**")
    print("```bash")
    print('export SKYROUTER_API_KEY="your-api-key"')
    print('export GENOPS_TEAM="development-team"')
    print('export GENOPS_PROJECT="skyrouter-dev"')
    print('export GENOPS_ENVIRONMENT="development"')
    print('export GENOPS_DAILY_BUDGET_LIMIT="50.0"')
    print('export GENOPS_GOVERNANCE_POLICY="advisory"')
    print('export SKYROUTER_ROUTING_STRATEGY="cost_optimized"')
    print("```")
    print()

    print("🚀 **Production Environment:**")
    print("```bash")
    print('export SKYROUTER_API_KEY="your-production-api-key"')
    print('export GENOPS_TEAM="ai-platform"')
    print('export GENOPS_PROJECT="multi-model-routing"')
    print('export GENOPS_ENVIRONMENT="production"')
    print('export GENOPS_DAILY_BUDGET_LIMIT="500.0"')
    print('export GENOPS_GOVERNANCE_POLICY="enforced"')
    print('export GENOPS_COST_CENTER="ai-operations"')
    print('export SKYROUTER_ROUTING_STRATEGY="balanced"')
    print("```")
    print()

    print("🏢 **Enterprise Environment:**")
    print("```bash")
    print('export SKYROUTER_API_KEY="your-enterprise-api-key"')
    print('export GENOPS_TEAM="enterprise-ai"')
    print('export GENOPS_PROJECT="multi-model-enterprise"')
    print('export GENOPS_ENVIRONMENT="production"')
    print('export GENOPS_CUSTOMER_ID="enterprise-customer-123"')
    print('export GENOPS_DAILY_BUDGET_LIMIT="1000.0"')
    print('export GENOPS_GOVERNANCE_POLICY="enforced"')
    print('export SKYROUTER_PREFERRED_MODELS="gpt-4,claude-3-opus,gemini-pro"')
    print('export SKYROUTER_ROUTING_STRATEGY="reliability_first"')
    print("```")


def test_basic_functionality():
    """Test basic SkyRouter adapter functionality."""

    print()
    print("🧪 Testing Basic Functionality")
    print("=" * 35)
    print()

    try:
        from genops.providers.skyrouter import GenOpsSkyRouterAdapter

        # Test adapter initialization
        print("🔧 Testing adapter initialization...")
        adapter = GenOpsSkyRouterAdapter(
            team="validation-test", project="setup-validation", daily_budget_limit=10.0
        )
        print("✅ Adapter initialized successfully")

        # Test session creation
        print("🔧 Testing session creation...")
        with adapter.track_routing_session("validation-test") as session:
            print("✅ Session created successfully")

            # Test cost calculation
            print("🔧 Testing cost calculation...")
            cost_result = session.track_model_call(
                model="gpt-3.5-turbo",
                input_data={"prompt": "Test validation"},
                cost=0.001,  # Provide explicit cost for testing
            )
            print(f"✅ Cost calculation successful: ${cost_result.total_cost:.3f}")

            # Test multi-model routing
            print("🔧 Testing multi-model routing...")
            route_result = session.track_multi_model_routing(
                models=["gpt-3.5-turbo", "claude-3-haiku"],
                input_data={"prompt": "Test multi-model routing"},
                routing_strategy="cost_optimized",
                cost=0.002,
            )
            print(f"✅ Multi-model routing successful: ${route_result.total_cost:.3f}")

        print()
        print("🎉 All functionality tests passed!")
        print("📊 Session summary:")
        print(f"   • Total cost: ${session.total_cost:.3f}")
        print(f"   • Operations: {session.operation_count}")
        print(f"   • Duration: {session.duration_seconds:.1f}s")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure GenOps is installed: pip install genops[skyrouter]")
        return False
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        print("💡 Check your configuration and try again")
        return False


def show_next_steps():
    """Show recommended next steps after validation."""

    print()
    print("🚀 Next Steps")
    print("=" * 15)
    print()

    print("Now that your setup is validated, try these examples:")
    print()
    print("1. **Basic Multi-Model Routing** (5 minutes)")
    print("   python basic_routing.py")
    print("   → Learn fundamental multi-model routing with governance")
    print()

    print("2. **Auto-Instrumentation** (3 minutes)")
    print("   python auto_instrumentation.py")
    print("   → See zero-code integration in action")
    print()

    print("3. **Route Optimization** (15 minutes)")
    print("   python route_optimization.py")
    print("   → Explore intelligent routing and cost optimization")
    print()

    print("4. **Agent Workflows** (20 minutes)")
    print("   python agent_workflows.py")
    print("   → Learn multi-agent routing patterns")
    print()

    print("5. **Enterprise Patterns** (30 minutes)")
    print("   python enterprise_patterns.py")
    print("   → Production deployment patterns")
    print()

    print("📚 **Documentation:**")
    print("   • Quickstart: docs/skyrouter-quickstart.md")
    print("   • Complete Guide: docs/integrations/skyrouter.md")
    print("   • Performance: docs/skyrouter-performance-benchmarks.md")
    print()

    print("💬 **Get Help:**")
    print("   • GitHub Discussions: https://github.com/KoshiHQ/GenOps-AI/discussions")
    print("   • Issues: https://github.com/KoshiHQ/GenOps-AI/issues")


def main():
    """Main execution function."""

    # Check for help flag
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print("SkyRouter + GenOps Setup Validation")
        print()
        print(
            "This script validates your SkyRouter + GenOps configuration for multi-model routing."
        )
        print()
        print("Usage:")
        print("  python setup_validation.py              # Run validation")
        print("  python setup_validation.py --help       # Show this help")
        print("  python setup_validation.py --examples   # Show configuration examples")
        print("  python setup_validation.py --test       # Run functionality tests")
        print()
        print("Environment Variables:")
        print("  SKYROUTER_API_KEY     - Your SkyRouter API key (required)")
        print("  GENOPS_TEAM           - Team name for cost attribution")
        print("  GENOPS_PROJECT        - Project name for cost attribution")
        print("  GENOPS_ENVIRONMENT    - Environment (development/staging/production)")
        return

    # Show configuration examples
    if len(sys.argv) > 1 and sys.argv[1] == "--examples":
        demonstrate_configuration_examples()
        return

    # Run functionality tests
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        if test_basic_functionality():
            show_next_steps()
        return

    # Run main validation
    try:
        validation_passed = run_validation()

        if validation_passed:
            # Run optional functionality test
            print()
            print("🧪 Would you like to test basic functionality? (y/n): ", end="")
            try:
                user_input = input().strip().lower()
                if user_input in ["y", "yes", ""]:
                    test_basic_functionality()
            except KeyboardInterrupt:
                print()

            show_next_steps()
        else:
            print()
            demonstrate_configuration_examples()

    except KeyboardInterrupt:
        print()
        print("👋 Validation cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        print()
        print("🔧 Troubleshooting tips:")
        print("1. Make sure you're in the project root directory")
        print("2. Check that GenOps is installed: pip install genops[skyrouter]")
        print("3. Verify your environment variables are set correctly")
        sys.exit(1)


if __name__ == "__main__":
    main()
