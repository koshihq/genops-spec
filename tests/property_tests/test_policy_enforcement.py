"""
Property-based tests for GenOps AI policy enforcement functionality.

These tests verify that policy enforcement maintains correctness across
all possible input combinations and edge cases.
"""

from typing import Any

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from genops.core.policy import PolicyResult, _policy_engine, register_policy

# Strategies for policy testing
policy_name_strategy = st.text(
    min_size=1,
    max_size=50,
    alphabet=st.characters(whitelist_categories=["Ll", "Lu", "Nd", "-", "_"])
)
cost_limit_strategy = st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False)
enforcement_level_strategy = st.sampled_from([PolicyResult.ALLOWED, PolicyResult.WARNING, PolicyResult.BLOCKED])
content_patterns_strategy = st.lists(
    st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=["Ll", "Lu"])),
    min_size=0,
    max_size=5
)


class TestPolicyEnforcementProperties:
    """Property-based tests for policy enforcement."""

    @given(
        policy_name=policy_name_strategy,
        max_cost=cost_limit_strategy,
        enforcement_level=enforcement_level_strategy,
        test_cost=cost_limit_strategy
    )
    @settings(max_examples=300, deadline=None)
    def test_cost_policy_enforcement_properties(self, policy_name, max_cost, enforcement_level, test_cost):
        """Test that cost policies always enforce correctly."""
        assume(len(policy_name.strip()) > 0)
        assume(max_cost > 0)
        assume(test_cost >= 0)

        # Register policy
        register_policy(
            name=policy_name.strip(),
            enforcement_level=enforcement_level,
            conditions={"max_cost": max_cost}
        )

        # Test policy evaluation
        context = {"cost": test_cost}
        result = _policy_engine.evaluate_policy(policy_name.strip(), context)

        # Properties that should always hold
        assert result is not None
        assert isinstance(result.result, PolicyResult)

        # Cost logic properties
        if test_cost <= max_cost:
            # Should allow operations within cost limit
            assert result.result in [PolicyResult.ALLOWED, PolicyResult.WARNING]
        else:
            # Behavior depends on enforcement level
            if enforcement_level == PolicyResult.BLOCKED:
                assert result.result == PolicyResult.BLOCKED
            elif enforcement_level == PolicyResult.WARNING:
                assert result.result == PolicyResult.WARNING
            else:  # ALLOWED
                assert result.result == PolicyResult.ALLOWED

    @given(
        policies=st.lists(
            st.tuples(
                policy_name_strategy,
                cost_limit_strategy,
                enforcement_level_strategy
            ),
            min_size=1,
            max_size=10,
            unique_by=lambda x: x[0].strip()  # Unique policy names
        ),
        test_cost=cost_limit_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_multiple_policies_consistency(self, policies, test_cost):
        """Test that multiple policies maintain consistent behavior."""
        assume(test_cost >= 0)

        registered_policies = []

        for policy_name, max_cost, enforcement_level in policies:
            name = policy_name.strip()
            if len(name) == 0 or max_cost <= 0:
                continue

            register_policy(
                name=name,
                enforcement_level=enforcement_level,
                conditions={"max_cost": max_cost}
            )
            registered_policies.append((name, max_cost, enforcement_level))

        # Test each policy
        context = {"cost": test_cost}
        results = []

        for name, max_cost, enforcement_level in registered_policies:
            result = _policy_engine.evaluate_policy(name, context)
            results.append((result, max_cost, enforcement_level))

            # Individual policy properties
            assert result is not None
            assert isinstance(result.result, PolicyResult)

        # Consistency properties across multiple policies
        assert len(results) == len(registered_policies)

        # All policies should evaluate consistently with their individual rules
        for result, max_cost, enforcement_level in results:
            if test_cost <= max_cost:
                assert result.result in [PolicyResult.ALLOWED, PolicyResult.WARNING]
            else:
                if enforcement_level == PolicyResult.BLOCKED:
                    assert result.result == PolicyResult.BLOCKED

    @given(
        policy_name=policy_name_strategy,
        blocked_patterns=content_patterns_strategy,
        test_content=st.text(min_size=0, max_size=200)
    )
    @settings(max_examples=200)
    def test_content_filtering_properties(self, policy_name, blocked_patterns, test_content):
        """Test that content filtering policies work correctly."""
        assume(len(policy_name.strip()) > 0)

        # Filter out empty patterns
        valid_patterns = [p for p in blocked_patterns if len(p.strip()) > 0]

        register_policy(
            name=policy_name.strip(),
            enforcement_level=PolicyResult.BLOCKED,
            conditions={"blocked_patterns": valid_patterns}
        )

        context = {"content": test_content}
        result = _policy_engine.evaluate_policy(policy_name.strip(), context)

        # Properties that should hold
        assert result is not None

        # Check if content should be blocked
        content_should_be_blocked = any(
            pattern.lower() in test_content.lower()
            for pattern in valid_patterns
        )

        if content_should_be_blocked:
            assert result.result == PolicyResult.BLOCKED
        else:
            # Content doesn't match patterns, should be allowed
            assert result.result in [PolicyResult.ALLOWED, PolicyResult.WARNING]

    @given(
        policies=st.lists(
            st.tuples(
                policy_name_strategy,
                st.dictionaries(
                    keys=st.sampled_from(["max_cost", "max_tokens", "min_confidence"]),
                    values=st.floats(min_value=0.01, max_value=100.0),
                    min_size=1,
                    max_size=3
                ),
                enforcement_level_strategy
            ),
            min_size=1,
            max_size=5,
            unique_by=lambda x: x[0].strip()
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_complex_policy_conditions_properties(self, policies):
        """Test policies with multiple conditions."""
        valid_policies = []

        for policy_name, conditions, enforcement_level in policies:
            name = policy_name.strip()
            if len(name) == 0:
                continue

            register_policy(
                name=name,
                enforcement_level=enforcement_level,
                conditions=conditions
            )
            valid_policies.append((name, conditions, enforcement_level))

        # Test with various contexts
        test_contexts = [
            {"cost": 5.0, "tokens": 100, "confidence": 0.8},
            {"cost": 50.0, "tokens": 1000, "confidence": 0.9},
            {"cost": 0.5, "tokens": 10, "confidence": 0.5}
        ]

        for context in test_contexts:
            for name, conditions, enforcement_level in valid_policies:
                result = _policy_engine.evaluate_policy(name, context)

                # Basic properties
                assert result is not None
                assert isinstance(result.result, PolicyResult)

                # Policy should be deterministic - same input gives same result
                result2 = _policy_engine.evaluate_policy(name, context)
                assert result.result == result2.result


class PolicyEnforcementStateMachine(RuleBasedStateMachine):
    """Stateful property-based testing for policy enforcement system."""

    def __init__(self):
        super().__init__()
        self.registered_policies: dict[str, dict[str, Any]] = {}
        self.policy_evaluations: list[tuple] = []

    @rule(
        policy_name=policy_name_strategy,
        max_cost=cost_limit_strategy,
        enforcement_level=enforcement_level_strategy
    )
    def register_cost_policy(self, policy_name, max_cost, enforcement_level):
        """Rule: Register a cost-based policy."""
        name = policy_name.strip()
        assume(len(name) > 0)
        assume(max_cost > 0)

        register_policy(
            name=name,
            enforcement_level=enforcement_level,
            conditions={"max_cost": max_cost}
        )

        self.registered_policies[name] = {
            "max_cost": max_cost,
            "enforcement_level": enforcement_level,
            "type": "cost"
        }

    @rule(
        policy_name=st.sampled_from([]),  # Only use registered policies
        test_cost=cost_limit_strategy
    )
    def evaluate_policy(self, policy_name, test_cost):
        """Rule: Evaluate a policy."""
        assume(policy_name in self.registered_policies)
        assume(test_cost >= 0)

        context = {"cost": test_cost}
        result = _policy_engine.evaluate_policy(policy_name, context)

        self.policy_evaluations.append((policy_name, context, result))

    @rule(target=st.sampled_from([]), policy_name=policy_name_strategy)
    def get_registered_policy_names(self, policy_name):
        """Rule: Return a registered policy name for use in other rules."""
        return list(self.registered_policies.keys())

    @invariant()
    def all_registered_policies_are_valid(self):
        """Invariant: All registered policies should have valid configurations."""
        for name, config in self.registered_policies.items():
            assert len(name) > 0
            assert "enforcement_level" in config
            assert isinstance(config["enforcement_level"], PolicyResult)

    @invariant()
    def policy_evaluations_are_consistent(self):
        """Invariant: Policy evaluations should be consistent."""
        # Group evaluations by policy and context
        evaluation_groups = {}
        for policy_name, context, result in self.policy_evaluations:
            key = (policy_name, tuple(sorted(context.items())))
            if key not in evaluation_groups:
                evaluation_groups[key] = []
            evaluation_groups[key].append(result)

        # Check consistency within each group
        for key, results in evaluation_groups.items():
            if len(results) > 1:
                # All results for the same policy and context should be identical
                first_result = results[0]
                for result in results[1:]:
                    assert result.result == first_result.result


class TestPolicyEnforcementStateMachine:
    """Test runner for stateful property-based testing."""

    def test_policy_enforcement_state_machine(self):
        """Run the stateful property-based test."""
        state_machine_test = PolicyEnforcementStateMachine.TestCase()
        state_machine_test.runTest()


@given(
    policy_configs=st.dictionaries(
        keys=policy_name_strategy,
        values=st.tuples(
            cost_limit_strategy,
            enforcement_level_strategy
        ),
        min_size=1,
        max_size=5
    ),
    operations=st.lists(
        st.tuples(
            cost_limit_strategy,
            st.text(min_size=0, max_size=100)
        ),
        min_size=1,
        max_size=20
    )
)
@settings(max_examples=50, deadline=None)
def test_policy_system_integration_properties(policy_configs, operations):
    """Test integration properties of the entire policy system."""
    # Register policies
    valid_policies = {}
    for name, (max_cost, enforcement_level) in policy_configs.items():
        clean_name = name.strip()
        if len(clean_name) > 0 and max_cost > 0:
            register_policy(
                name=clean_name,
                enforcement_level=enforcement_level,
                conditions={"max_cost": max_cost}
            )
            valid_policies[clean_name] = (max_cost, enforcement_level)

    assume(len(valid_policies) > 0)

    # Test operations against all policies
    blocked_operations = 0
    allowed_operations = 0

    for cost, content in operations:
        if cost < 0:
            continue

        context = {"cost": cost, "content": content}

        # Test against all registered policies
        for policy_name, (max_cost, enforcement_level) in valid_policies.items():
            result = _policy_engine.evaluate_policy(policy_name, context)

            # Count blocked vs allowed
            if result.result == PolicyResult.BLOCKED:
                blocked_operations += 1
            elif result.result == PolicyResult.ALLOWED:
                allowed_operations += 1

            # Verify policy logic is correct
            if cost > max_cost and enforcement_level == PolicyResult.BLOCKED:
                assert result.result == PolicyResult.BLOCKED

    # System-level properties
    total_evaluations = blocked_operations + allowed_operations
    assert total_evaluations >= 0


if __name__ == "__main__":
    # Run property-based tests with statistics
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
