#!/usr/bin/env python3
"""Quick test of the new attribution system."""

import genops


def test_attribution_system():
    """Test the new attribution context system."""
    print("🧪 Testing GenOps AI Attribution System")
    print("=" * 50)

    # Test 1: Global defaults
    print("\n1. Testing global defaults...")
    genops.set_default_attributes(
        team="test-team",
        project="test-project",
        environment="testing"
    )

    defaults = genops.get_default_attributes()
    print(f"   Defaults: {defaults}")
    assert defaults["team"] == "test-team"
    assert defaults["project"] == "test-project"
    assert defaults["environment"] == "testing"
    print("   ✅ Global defaults working")

    # Test 2: Context attributes
    print("\n2. Testing context attributes...")
    genops.set_context(
        customer_id="test-customer",
        user_id="test-user"
    )

    context = genops.get_context()
    print(f"   Context: {context}")
    assert context["customer_id"] == "test-customer"
    assert context["user_id"] == "test-user"
    print("   ✅ Context attributes working")

    # Test 3: Effective attributes with priority
    print("\n3. Testing effective attributes priority...")
    effective = genops.get_effective_attributes(
        team="override-team",  # Should override default
        feature="test-feature"  # New attribute
    )
    print(f"   Effective: {effective}")

    # Check priority: operation > context > defaults
    assert effective["team"] == "override-team"  # Operation override
    assert effective["project"] == "test-project"  # From defaults
    assert effective["customer_id"] == "test-customer"  # From context
    assert effective["feature"] == "test-feature"  # Operation-specific
    print("   ✅ Priority hierarchy working correctly")

    # Test 4: Convenience functions
    print("\n4. Testing convenience functions...")
    genops.set_team_defaults(
        team="convenience-team",
        project="convenience-project",
        cost_center="engineering"
    )

    genops.set_customer_context(
        customer_id="enterprise-123",
        customer_name="Acme Corp",
        tier="enterprise"
    )

    final_effective = genops.get_effective_attributes(feature="final-test")
    print(f"   Final effective: {final_effective}")

    assert final_effective["team"] == "convenience-team"
    assert final_effective["customer_id"] == "enterprise-123"
    assert final_effective["feature"] == "final-test"
    print("   ✅ Convenience functions working")

    # Test 5: Clear functions
    print("\n5. Testing clear functions...")
    genops.clear_context()
    genops.clear_default_attributes()

    assert genops.get_context() == {}
    assert genops.get_default_attributes() == {}
    print("   ✅ Clear functions working")

    print("\n🎉 ALL ATTRIBUTION TESTS PASSED!")
    print("The new attribution system is ready for use!")

if __name__ == "__main__":
    test_attribution_system()
