#!/usr/bin/env python3
"""
🛡️ Tag Validation and Enforcement Guide for GenOps AI

This example demonstrates how to validate attribution tags to ensure
data quality, compliance, and consistency across your AI operations.

VALIDATION CAPABILITIES:
✅ Required fields enforcement
✅ Format pattern validation (regex)
✅ Enum value constraints
✅ Length limits and custom rules
✅ Configurable severity levels (warning, error, block)
✅ Custom validation functions

Run this example to see all tag validation patterns in action!
"""

import logging
import genops
from genops import (
    ValidationSeverity, 
    ValidationRule,
    TagValidationError,
    validate_tags,
    enforce_tags,
    add_validation_rule,
    get_validator,
    create_required_rule,
    create_enum_rule,
    create_pattern_rule
)

# Set up logging to see validation messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def demonstrate_basic_validation():
    """Show basic tag validation with default rules."""
    print("\n🛡️ BASIC TAG VALIDATION")
    print("=" * 60)
    
    print("🔍 Testing with good attributes...")
    good_attributes = {
        "team": "platform-engineering",
        "project": "ai-services", 
        "customer_id": "enterprise-123",
        "environment": "production",
        "feature": "chat-assistant",
        "user_id": "user_456"
    }
    
    result = validate_tags(good_attributes)
    print(f"✅ Valid: {result.valid}")
    print(f"📊 Warnings: {len(result.warnings)}")
    print(f"❌ Violations: {len(result.violations)}")
    
    if result.warnings:
        for warning in result.warnings:
            print(f"   ⚠️ {warning['message']}")
    
    print("\n🔍 Testing with problematic attributes...")
    bad_attributes = {
        "team": "Platform Engineering",  # Wrong format (spaces)
        "customer_id": "enterprise@123",  # Invalid characters
        "environment": "dev",            # Not in allowed enum
        "feature": "x",                  # Too short
        "user_id": ""                    # Empty string
    }
    
    result = validate_tags(bad_attributes)
    print(f"✅ Valid: {result.valid}")
    print(f"📊 Warnings: {len(result.warnings)}")
    print(f"❌ Violations: {len(result.violations)}")
    
    for warning in result.warnings:
        print(f"   ⚠️ WARNING: {warning['message']}")
    
    for violation in result.violations:
        print(f"   ❌ ERROR: {violation['message']}")


def demonstrate_custom_validation_rules():
    """Show how to add custom validation rules."""
    print("\n⚙️ CUSTOM VALIDATION RULES")
    print("=" * 60)
    
    validator = get_validator()
    
    # Add custom required field
    validator.add_rule(create_required_rule(
        "cost_center", 
        severity=ValidationSeverity.ERROR
    ))
    
    # Add custom enum for customer tiers
    validator.add_rule(create_enum_rule(
        "customer_tier",
        allowed_values={"freemium", "startup", "enterprise", "enterprise-plus"},
        severity=ValidationSeverity.WARNING
    ))
    
    # Add custom pattern for API keys
    validator.add_rule(create_pattern_rule(
        "api_key",
        pattern=r"^ak_[a-z]+_[a-zA-Z0-9]{32}$",
        description="API keys must follow format: ak_env_32chars",
        severity=ValidationSeverity.ERROR
    ))
    
    # Add custom validation function
    def validate_budget_amount(value):
        """Custom validator for budget amounts."""
        if value is None:
            return True  # Optional field
        try:
            amount = float(value)
            return 0 < amount <= 1000000  # $0-$1M range
        except (ValueError, TypeError):
            return False
    
    validator.add_rule(ValidationRule(
        name="budget_amount_range",
        attribute="budget_amount",
        rule_type="custom",
        severity=ValidationSeverity.WARNING,
        description="Budget amount must be between $0-$1M",
        validator_func=validate_budget_amount,
        error_message="Budget amount must be a number between $0.01 and $1,000,000"
    ))
    
    print("✅ Added custom validation rules:")
    print("   • cost_center (required)")
    print("   • customer_tier (enum)")
    print("   • api_key (pattern)")
    print("   • budget_amount (custom function)")
    
    print("\n🔍 Testing custom rules...")
    test_attributes = {
        "team": "platform-engineering",
        "cost_center": "engineering",  # Required and present
        "customer_tier": "premium",    # Invalid enum value
        "api_key": "invalid-key",      # Wrong pattern
        "budget_amount": "1500000"     # Too high
    }
    
    result = validate_tags(test_attributes)
    print(f"✅ Valid: {result.valid}")
    
    for warning in result.warnings:
        print(f"   ⚠️ WARNING: {warning['message']}")
    
    for violation in result.violations:
        print(f"   ❌ ERROR: {violation['message']}")


def demonstrate_severity_levels():
    """Show different validation severity levels in action."""
    print("\n📊 VALIDATION SEVERITY LEVELS")
    print("=" * 60)
    
    validator = get_validator()
    
    # Clear existing rules and add test rules with different severities
    validator.rules.clear()
    
    # WARNING: Logs warning, allows operation
    validator.add_rule(ValidationRule(
        name="team_format_warning",
        attribute="team",
        rule_type="pattern",
        severity=ValidationSeverity.WARNING,
        description="Team format warning",
        pattern=r"^[a-z-]+$",
        error_message="Team should be lowercase with hyphens"
    ))
    
    # ERROR: Logs error, allows operation but marks invalid
    validator.add_rule(ValidationRule(
        name="environment_required_error", 
        attribute="environment",
        rule_type="required",
        severity=ValidationSeverity.ERROR,
        description="Environment is required",
        error_message="Environment must be specified"
    ))
    
    # BLOCK: Raises exception, prevents operation
    validator.add_rule(ValidationRule(
        name="customer_id_required_block",
        attribute="customer_id",
        rule_type="required",
        severity=ValidationSeverity.BLOCK,
        description="Customer ID is required for billing",
        error_message="Customer ID is required and cannot be empty"
    ))
    
    print("🔍 Testing WARNING level (team format)...")
    result = validate_tags({"team": "Platform_Engineering"})
    print(f"   Valid: {result.valid} (operation continues)")
    print(f"   Warnings: {len(result.warnings)}")
    
    print("\n🔍 Testing ERROR level (missing environment)...")
    result = validate_tags({"team": "platform-eng"})
    print(f"   Valid: {result.valid} (operation continues)")
    print(f"   Violations: {len(result.violations)}")
    
    print("\n🔍 Testing BLOCK level (missing customer_id)...")
    try:
        enforce_tags({"team": "platform-eng", "environment": "prod"})
        print("   ✅ Operation allowed")
    except TagValidationError as e:
        print(f"   🚫 BLOCKED: {e}")
        print(f"   📋 Violations: {len(e.violations)}")


def demonstrate_integration_with_providers():
    """Show validation integrated with AI provider operations."""  
    print("\n🤖 PROVIDER INTEGRATION WITH VALIDATION")
    print("=" * 60)
    
    # Set up validation rules for production use
    validator = get_validator()
    validator.rules.clear()  # Start fresh
    
    # Production validation rules
    validator.add_rule(create_required_rule("team", ValidationSeverity.WARNING))
    validator.add_rule(create_required_rule("customer_id", ValidationSeverity.ERROR))
    
    validator.add_rule(create_enum_rule(
        "environment",
        {"production", "staging", "development"},
        ValidationSeverity.WARNING
    ))
    
    # Set up some defaults with validation issues
    genops.set_default_attributes(
        team="platform-engineering",  # Good
        environment="dev",             # Warning - not in enum
        project="ai-services"
    )
    
    print("🏷️ Set defaults with validation warnings...")
    print("   team: platform-engineering ✅")
    print("   environment: dev ⚠️ (should be 'development')")
    print("   project: ai-services ✅")
    
    print("\n🔍 Getting effective attributes for operation...")
    try:
        # This will trigger validation
        effective_attrs = genops.get_effective_attributes(
            customer_id="enterprise-123",  # Required field provided
            feature="chat-assistant"
        )
        
        print("✅ Validation passed, effective attributes:")
        for key, value in sorted(effective_attrs.items()):
            print(f"   {key}: {value}")
            
    except TagValidationError as e:
        print(f"🚫 Validation blocked operation: {e}")
    
    print("\n🔍 Testing with missing required customer_id...")
    try:
        effective_attrs = genops.get_effective_attributes(
            feature="chat-assistant"
            # customer_id missing - should trigger ERROR
        )
        
        print("⚠️ Validation errors logged, but operation continues")
        print("✅ Effective attributes still generated")
        
    except TagValidationError as e:
        print(f"🚫 Operation blocked: {e}")


def demonstrate_enterprise_compliance_rules():
    """Show enterprise-grade validation rules for compliance."""
    print("\n🏢 ENTERPRISE COMPLIANCE VALIDATION") 
    print("=" * 60)
    
    validator = get_validator()
    validator.rules.clear()
    
    # Compliance rules for enterprise
    
    # 1. Data residency compliance
    validator.add_rule(create_enum_rule(
        "data_region",
        {"us-east", "us-west", "eu-central", "ap-southeast"},
        ValidationSeverity.ERROR
    ))
    
    # 2. Cost center required for FinOps
    validator.add_rule(create_required_rule("cost_center", ValidationSeverity.BLOCK))
    
    # 3. Customer classification
    validator.add_rule(create_enum_rule(
        "customer_classification", 
        {"public", "confidential", "restricted", "top-secret"},
        ValidationSeverity.BLOCK
    ))
    
    # 4. Compliance tags required
    validator.add_rule(create_required_rule("compliance_scope", ValidationSeverity.ERROR))
    
    # 5. Custom PII detection
    def validate_no_pii_in_feature(value):
        """Ensure feature names don't contain PII."""
        if value is None:
            return True
        pii_patterns = ['email', 'phone', 'ssn', 'credit_card', 'personal']
        return not any(pattern in str(value).lower() for pattern in pii_patterns)
    
    validator.add_rule(ValidationRule(
        name="feature_no_pii",
        attribute="feature",
        rule_type="custom", 
        severity=ValidationSeverity.BLOCK,
        description="Feature names must not contain PII indicators",
        validator_func=validate_no_pii_in_feature,
        error_message="Feature name contains potential PII - use generic names"
    ))
    
    print("🛡️ Enterprise compliance rules configured:")
    print("   • Data residency validation")
    print("   • Cost center required") 
    print("   • Customer classification required")
    print("   • Compliance scope required")
    print("   • PII detection in feature names")
    
    print("\n✅ Testing compliant attributes...")
    compliant_attrs = {
        "team": "platform-engineering",
        "cost_center": "engineering",
        "data_region": "us-east",
        "customer_classification": "confidential",
        "compliance_scope": "sox-compliant",
        "feature": "document-processing"
    }
    
    try:
        result = enforce_tags(compliant_attrs)
        print("   ✅ All compliance rules passed!")
        
    except TagValidationError as e:
        print(f"   🚫 Compliance violation: {e}")
    
    print("\n❌ Testing non-compliant attributes...")
    non_compliant_attrs = {
        "team": "platform-engineering",
        # cost_center missing (BLOCK)
        "data_region": "china",  # Invalid region 
        "customer_classification": "internal",  # Invalid classification
        "feature": "email-processing"  # Contains PII indicator
    }
    
    try:
        result = enforce_tags(non_compliant_attrs)
        print("   ⚠️ Unexpected success - should have blocked")
        
    except TagValidationError as e:
        print(f"   🚫 Blocked as expected: {e}")
        print(f"   📋 Total violations: {len(e.violations)}")


def demonstrate_configuration_management():
    """Show validation configuration and management."""
    print("\n⚙️ VALIDATION CONFIGURATION MANAGEMENT")
    print("=" * 60)
    
    validator = get_validator()
    
    print("🔧 Current validation status:")
    print(f"   Enabled: {validator.enabled}")
    print(f"   Rules count: {len(validator.rules)}")
    
    print("\n📋 Current validation rules:")
    for rule in validator.rules:
        print(f"   • {rule.name} ({rule.severity.value}) - {rule.description}")
    
    print("\n⏸️ Disabling validation temporarily...")
    validator.disable()
    
    # Test with bad data - should pass
    bad_data = {"team": "INVALID FORMAT", "environment": "invalid"}
    result = validate_tags(bad_data)
    print(f"   With validation disabled - Valid: {result.valid}")
    print(f"   Warnings: {len(result.warnings)}, Violations: {len(result.violations)}")
    
    print("\n▶️ Re-enabling validation...")
    validator.enable()
    
    result = validate_tags(bad_data)
    print(f"   With validation enabled - Valid: {result.valid}")
    print(f"   Warnings: {len(result.warnings)}, Violations: {len(result.violations)}")
    
    print("\n🧹 Cleaning up rules...")
    initial_count = len(validator.rules)
    
    # Remove specific rules
    validator.remove_rule("feature_no_pii")
    validator.remove_rule("customer_classification_enum")
    
    print(f"   Rules before: {initial_count}")
    print(f"   Rules after: {len(validator.rules)}")


def main():
    """Run the complete tag validation and enforcement demonstration."""
    print("🛡️ GenOps AI: Tag Validation and Enforcement Guide")
    print("=" * 80)
    print("\nThis guide demonstrates how to validate attribution tags for")
    print("data quality, compliance, and consistency across AI operations.")
    
    # Run all demonstrations
    demonstrate_basic_validation()
    demonstrate_custom_validation_rules()
    demonstrate_severity_levels()
    demonstrate_integration_with_providers()
    demonstrate_enterprise_compliance_rules()
    demonstrate_configuration_management()
    
    print(f"\n🎯 KEY TAKEAWAYS")
    print("=" * 60)
    print("✅ Default validation rules ensure basic data quality")
    print("✅ Custom rules support enterprise compliance requirements")
    print("✅ Three severity levels: WARNING, ERROR, BLOCK")
    print("✅ Automatic integration with attribution system")
    print("✅ Configurable and extensible validation framework")
    print("✅ PII detection and enterprise governance support")
    
    print(f"\n📚 NEXT STEPS")
    print("=" * 60)
    print("1. Configure validation rules for your organization's needs")
    print("2. Set up enterprise compliance rules (data residency, PII, etc.)")
    print("3. Integrate with your CI/CD pipeline for automated validation")
    print("4. Monitor validation metrics in your observability platform")
    print("5. Train teams on proper attribution tag formats and requirements")
    
    print(f"\n🔗 Learn more: https://docs.genopsai.org/validation")


if __name__ == "__main__":
    main()