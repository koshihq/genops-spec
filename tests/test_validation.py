#!/usr/bin/env python3
"""Test the tag validation and enforcement system."""

import genops
from genops import (
    TagValidationError,
    ValidationRule,
    ValidationSeverity,
    create_enum_rule,
    create_pattern_rule,
    create_required_rule,
    enforce_tags,
    get_validator,
    validate_tags,
)


def test_basic_validation():
    """Test basic validation functionality."""
    print("🧪 Testing basic tag validation...")

    # Test valid attributes
    good_attrs = {
        "team": "platform-engineering",
        "environment": "production",
        "customer_id": "enterprise-123"
    }

    result = validate_tags(good_attrs)
    assert result.valid
    print("   ✅ Valid attributes pass validation")

    # Test invalid attributes
    bad_attrs = {
        "team": "Invalid Team Name",  # Wrong format
        "environment": "invalid",     # Not in enum
        "customer_id": "bad@id",      # Invalid characters
        "user_id": ""                 # Empty string
    }

    result = validate_tags(bad_attrs)
    assert len(result.violations) > 0 or len(result.warnings) > 0
    print("   ✅ Invalid attributes trigger violations/warnings")


def test_severity_levels():
    """Test different validation severity levels."""
    print("🧪 Testing validation severity levels...")

    validator = get_validator()
    validator.rules.clear()

    # Add rules with different severities
    validator.add_rule(ValidationRule(
        name="test_warning",
        attribute="test_attr",
        rule_type="required",
        severity=ValidationSeverity.WARNING,
        description="Test warning"
    ))

    validator.add_rule(ValidationRule(
        name="test_error",
        attribute="test_attr2",
        rule_type="required",
        severity=ValidationSeverity.ERROR,
        description="Test error"
    ))

    validator.add_rule(ValidationRule(
        name="test_block",
        attribute="test_attr3",
        rule_type="required",
        severity=ValidationSeverity.BLOCK,
        description="Test block"
    ))

    # Test all severity levels at once
    result = validate_tags({})
    assert not result.valid  # Has blocking violations
    assert len(result.warnings) > 0  # Has warnings
    assert len(result.violations) > 0  # Has violations

    # Find specific severity types
    warnings = [v for v in result.warnings if v.get("severity") == "warning"]
    errors = [v for v in result.violations if v.get("severity") == "error"]
    blocks = [v for v in result.violations if v.get("severity") == "block"]

    assert len(warnings) > 0
    assert len(errors) > 0
    assert len(blocks) > 0
    print("   ✅ All severity levels work correctly")

    # Test BLOCK - should raise exception
    try:
        enforce_tags({})
        assert False, "Expected TagValidationError"
    except TagValidationError:
        print("   ✅ BLOCK severity prevents operation")


def test_custom_rules():
    """Test custom validation rules."""
    print("🧪 Testing custom validation rules...")

    validator = get_validator()
    validator.rules.clear()

    # Test pattern rule
    validator.add_rule(create_pattern_rule(
        "api_key",
        r"^ak_[a-z]+_[a-zA-Z0-9]{10}$",
        "API key format validation"
    ))

    valid_key = {"api_key": "ak_prod_abc1234567"}
    invalid_key = {"api_key": "invalid-key"}

    assert validate_tags(valid_key).valid
    assert len(validate_tags(invalid_key).violations + validate_tags(invalid_key).warnings) > 0
    print("   ✅ Pattern validation works")

    # Test enum rule
    validator.add_rule(create_enum_rule(
        "tier",
        {"free", "pro", "enterprise"}
    ))

    valid_tier = {"tier": "enterprise"}
    invalid_tier = {"tier": "premium"}

    assert validate_tags(valid_tier).valid
    assert len(validate_tags(invalid_tier).violations + validate_tags(invalid_tier).warnings) > 0
    print("   ✅ Enum validation works")


def test_context_integration():
    """Test integration with attribution context system."""
    print("🧪 Testing validation integration with context...")

    # Clear and set up validation
    validator = get_validator()
    validator.rules.clear()
    validator.add_rule(create_required_rule("team", ValidationSeverity.WARNING))

    # Clear context
    genops.clear_default_attributes()
    genops.clear_context()

    # Set defaults that should trigger validation
    genops.set_default_attributes(team="platform-engineering")

    # Should work without issues
    effective = genops.get_effective_attributes(customer_id="test-123")
    assert "team" in effective
    assert effective["team"] == "platform-engineering"
    print("   ✅ Valid defaults work with validation")

    # Test with validation that would trigger warnings
    genops.set_default_attributes(team="")  # Empty team - should warn

    effective = genops.get_effective_attributes(customer_id="test-123")
    # Should still work but log warning
    print("   ✅ Validation warnings don't break attribution")


def test_enable_disable():
    """Test enabling/disabling validation."""
    print("🧪 Testing validation enable/disable...")

    validator = get_validator()
    validator.rules.clear()
    validator.add_rule(create_required_rule("required_field", ValidationSeverity.BLOCK))

    # With validation enabled, should block
    try:
        enforce_tags({})
        assert False, "Should have been blocked"
    except TagValidationError:
        print("   ✅ Validation blocks when enabled")

    # Disable validation
    validator.disable()

    # Should pass now
    result = enforce_tags({})
    assert isinstance(result, dict)
    print("   ✅ Validation allows when disabled")

    # Re-enable
    validator.enable()

    # Should block again
    try:
        enforce_tags({})
        assert False, "Should have been blocked after re-enabling"
    except TagValidationError:
        print("   ✅ Validation blocks when re-enabled")


def main():
    """Run all validation tests."""
    print("🛡️ Testing GenOps AI Tag Validation System")
    print("=" * 60)

    try:
        test_basic_validation()
        test_severity_levels()
        test_custom_rules()
        test_context_integration()
        test_enable_disable()

        print("\n🎉 ALL VALIDATION TESTS PASSED!")
        print("The tag validation and enforcement system is working correctly!")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise

    finally:
        # Clean up
        genops.clear_default_attributes()
        genops.clear_context()
        get_validator().rules.clear()
        get_validator().enable()


if __name__ == "__main__":
    main()
