"""
Setup Validation Example

This example demonstrates how to validate your Kubetorch setup
and diagnose configuration issues.

Time to run: < 30 seconds
"""

from genops.providers.kubetorch import (
    validate_kubetorch_setup,
    print_validation_result,
    get_module_status,
)

print("=" * 60)
print("GenOps Kubetorch - Setup Validation")
print("=" * 60)

# =============================================
# Example 1: Quick Module Status Check
# =============================================
print("\n1. Module Status Check")
print("-" * 60)

status = get_module_status()

print("Module Availability:")
for module, available in status.items():
    status_icon = "✅" if available else "❌"
    print(f"  {status_icon} {module:20s}: {'Available' if available else 'Not Available'}")

# =============================================
# Example 2: Comprehensive Validation
# =============================================
print("\n2. Comprehensive Setup Validation")
print("-" * 60)

result = validate_kubetorch_setup()

# Print full validation report
print_validation_result(result, show_all=True)

# =============================================
# Example 3: Programmatic Validation Checks
# =============================================
print("\n3. Programmatic Validation Checks")
print("-" * 60)

if result.is_valid():
    print("✅ Setup is valid - ready to use!")
else:
    print(f"❌ Setup has {result.errors} error(s)")
    print("\nErrors found:")
    for issue in result.issues:
        if issue.level.value == "error":
            print(f"  - {issue.message}")
            if issue.fix_suggestion:
                print(f"    Fix: {issue.fix_suggestion}")

print(f"\nValidation Summary:")
print(f"  Total Checks: {result.total_checks}")
print(f"  ✅ Successful: {result.successful_checks}")
print(f"  ⚠️  Warnings: {result.warnings}")
print(f"  ❌ Errors: {result.errors}")

print("\n" + "=" * 60)
