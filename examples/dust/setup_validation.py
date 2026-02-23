#!/usr/bin/env python3
"""
Dust integration setup validation.

This example demonstrates:
- Comprehensive setup validation
- Environment variable checking
- API connectivity testing
- Workspace access verification
- Troubleshooting guidance

Prerequisites:
- pip install genops[dust]
- Set DUST_API_KEY and DUST_WORKSPACE_ID (optional for some checks)
"""

import os

from genops.providers.dust_validation import (
    check_dependencies,
    check_dust_connectivity,
    check_environment_variables,
    check_workspace_access,
    print_validation_result,
    quick_validate,
    validate_setup,
)


def main():
    """Comprehensive validation of Dust integration setup."""

    print("🔍 Dust Integration Setup Validation")
    print("=" * 50)

    # Quick validation check
    print("\n⚡ Quick Validation Check")
    print("-" * 30)

    if quick_validate():
        print("✅ Quick validation passed!")
    else:
        print("❌ Quick validation failed - running detailed checks...")

    # Comprehensive validation
    print("\n🔎 Comprehensive Validation")
    print("-" * 30)

    # Option 1: Validate with environment variables
    result = validate_setup()
    print_validation_result(result)

    # Option 2: Validate with explicit credentials (if available)
    api_key = os.getenv("DUST_API_KEY")
    workspace_id = os.getenv("DUST_WORKSPACE_ID")

    if api_key and workspace_id:
        print("\n🔐 Validating with Explicit Credentials")
        print("-" * 40)

        explicit_result = validate_setup(
            api_key=api_key, workspace_id=workspace_id, base_url="https://dust.tt"
        )

        print(
            f"Explicit validation result: {'✅ PASSED' if explicit_result.is_valid else '❌ FAILED'}"
        )

        if not explicit_result.is_valid:
            print("Issues found:")
            for issue in explicit_result.issues:
                if issue.level == "error":
                    print(f"  ❌ {issue.message}")
                elif issue.level == "warning":
                    print(f"  ⚠️  {issue.message}")

    # Individual component checks
    print("\n🧩 Individual Component Validation")
    print("-" * 40)

    # Environment variables
    print("\n📋 Environment Variables:")
    env_issues = check_environment_variables()
    for issue in env_issues:
        icon = (
            "❌" if issue.level == "error" else "⚠️" if issue.level == "warning" else "ℹ️"
        )
        print(f"  {icon} {issue.message}")
        if issue.fix_suggestion:
            print(f"      💡 {issue.fix_suggestion}")

    # Dependencies
    print("\n📦 Dependencies:")
    dep_issues = check_dependencies()
    for issue in dep_issues:
        icon = (
            "❌" if issue.level == "error" else "⚠️" if issue.level == "warning" else "ℹ️"
        )
        print(f"  {icon} {issue.message}")
        if issue.fix_suggestion:
            print(f"      💡 {issue.fix_suggestion}")

    # Connectivity (if credentials available)
    if api_key and workspace_id:
        print("\n🌐 API Connectivity:")
        conn_issues = check_dust_connectivity(api_key, workspace_id)
        for issue in conn_issues:
            icon = (
                "❌"
                if issue.level == "error"
                else "⚠️"
                if issue.level == "warning"
                else "ℹ️"
            )
            print(f"  {icon} {issue.message}")
            if issue.fix_suggestion:
                print(f"      💡 {issue.fix_suggestion}")

        print("\n🏢 Workspace Access:")
        access_issues = check_workspace_access(api_key, workspace_id)
        for issue in access_issues:
            icon = (
                "❌"
                if issue.level == "error"
                else "⚠️"
                if issue.level == "warning"
                else "ℹ️"
            )
            print(f"  {icon} {issue.message}")
            if issue.fix_suggestion:
                print(f"      💡 {issue.fix_suggestion}")

    # Configuration recommendations
    print("\n📝 Configuration Recommendations")
    print("-" * 40)

    recommendations = generate_setup_recommendations(
        result if "result" in locals() else None
    )
    for rec in recommendations:
        print(f"  • {rec}")

    # Next steps
    print("\n🚀 Next Steps")
    print("-" * 15)

    if result.is_valid:
        print("✅ Your Dust integration is ready!")
        print("  • Run 'python basic_tracking.py' to test basic operations")
        print("  • Try 'python cost_optimization.py' for cost analysis")
        print("  • Check 'python production_patterns.py' for enterprise setup")
    else:
        error_count = len([i for i in result.issues if i.level == "error"])
        print(f"❌ Fix {error_count} error(s) before proceeding")
        print("  • Review the issues above and apply suggested fixes")
        print("  • Re-run this validation after making changes")
        print("  • Check the Dust documentation for additional help")


def generate_setup_recommendations(result=None):
    """Generate personalized setup recommendations."""
    recommendations = []

    # Basic recommendations
    if not os.getenv("OTEL_SERVICE_NAME"):
        recommendations.append(
            "Set OTEL_SERVICE_NAME for better trace identification: "
            "export OTEL_SERVICE_NAME='my-dust-app'"
        )

    if not os.getenv("GENOPS_TEAM"):
        recommendations.append(
            "Set GENOPS_TEAM for cost attribution: export GENOPS_TEAM='ai-team'"
        )

    if not os.getenv("GENOPS_PROJECT"):
        recommendations.append(
            "Set GENOPS_PROJECT for project tracking: "
            "export GENOPS_PROJECT='customer-support'"
        )

    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        recommendations.append(
            "Configure OTLP endpoint for telemetry export: "
            "export OTEL_EXPORTER_OTLP_ENDPOINT='http://localhost:4317'"
        )

    # Environment-specific recommendations
    env = os.getenv("GENOPS_ENVIRONMENT", "").lower()
    if env not in ["development", "staging", "production"]:
        recommendations.append(
            "Set GENOPS_ENVIRONMENT for proper governance: "
            "export GENOPS_ENVIRONMENT='production'"
        )

    # Result-based recommendations
    if result and hasattr(result, "summary"):
        summary = result.summary

        if not summary.get("telemetry_configured", False):
            recommendations.append(
                "Configure OpenTelemetry for comprehensive observability"
            )

        if not summary.get("governance_attributes_configured", False):
            recommendations.append(
                "Set governance attributes (GENOPS_TEAM, GENOPS_PROJECT) for cost attribution"
            )

        if summary.get("warnings", 0) > 3:
            recommendations.append(
                "Consider addressing warnings to optimize your setup"
            )

    # Security recommendations
    if os.getenv("DUST_API_KEY") and not os.getenv("DUST_API_KEY").startswith("dust_"):
        recommendations.append(
            "Verify your API key format - Dust keys typically start with 'dust_'"
        )

    return recommendations


def demo_validation_in_code():
    """Demonstrate how to use validation in your application code."""

    print("\n🔧 Using Validation in Your Code")
    print("-" * 35)

    print("Example 1: Startup validation")
    print("""
def initialize_dust_service():
    from genops.providers.dust_validation import validate_setup

    result = validate_setup()
    if not result.is_valid:
        logger.error("Dust setup validation failed")
        for issue in result.issues:
            if issue.level == "error":
                logger.error(f"Setup error: {issue.message}")
        raise ValueError("Invalid Dust configuration")

    logger.info("Dust integration validated successfully")
    return instrument_dust()
""")

    print("Example 2: Health check endpoint")
    print("""
@app.route("/health/dust")
def dust_health_check():
    from genops.providers.dust_validation import quick_validate

    if quick_validate():
        return {"status": "healthy", "service": "dust"}
    else:
        return {"status": "unhealthy", "service": "dust"}, 503
""")

    print("Example 3: Configuration validation")
    print("""
def validate_dust_config(config):
    from genops.providers.dust_validation import validate_setup

    result = validate_setup(
        api_key=config.get("dust_api_key"),
        workspace_id=config.get("dust_workspace_id")
    )

    return {
        "valid": result.is_valid,
        "issues": [{"level": i.level, "message": i.message}
                   for i in result.issues],
        "summary": result.summary
    }
""")


if __name__ == "__main__":
    main()
    demo_validation_in_code()
