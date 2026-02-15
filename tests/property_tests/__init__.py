"""
Property-based tests for GenOps AI.

This package contains property-based tests using Hypothesis to automatically
generate thousands of test cases and catch edge cases that manual unit tests
might miss.

Property-based testing is particularly valuable for GenOps AI because:

1. **Cost Attribution**: Ensures mathematical properties hold across all possible
   cost values, currencies, and provider combinations.

2. **Policy Enforcement**: Verifies that policies behave correctly for all
   possible input combinations and edge cases.

3. **Telemetry Tracking**: Confirms that telemetry data maintains consistency
   and correctness regardless of operation complexity.

4. **Provider Integration**: Tests that provider adapters handle all possible
   API responses and error conditions correctly.

These tests complement traditional unit tests by exploring the vast input space
automatically and finding bugs that would be nearly impossible to discover
through manual test case creation.
"""

# Property-based testing configuration
HYPOTHESIS_SETTINGS = {
    "max_examples": 500,  # Run more examples for thorough testing
    "deadline": None,  # No time limit for complex property verification
    "derandomize": True,  # Consistent test runs
}

# Test categories
COST_ATTRIBUTION_TESTS = ["test_cost_attribution.py"]

POLICY_ENFORCEMENT_TESTS = ["test_policy_enforcement.py"]

TELEMETRY_TESTS = [
    # Future: test_telemetry_properties.py
]

PROVIDER_INTEGRATION_TESTS = [
    # Future: test_provider_properties.py
]

__all__ = [
    "HYPOTHESIS_SETTINGS",
    "COST_ATTRIBUTION_TESTS",
    "POLICY_ENFORCEMENT_TESTS",
    "TELEMETRY_TESTS",
    "PROVIDER_INTEGRATION_TESTS",
]
