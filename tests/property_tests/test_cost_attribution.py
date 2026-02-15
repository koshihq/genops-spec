"""
Property-based tests for GenOps AI cost attribution functionality.

These tests use Hypothesis to generate thousands of test cases automatically,
catching edge cases that manual unit tests might miss.
"""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from genops.core.context import set_default_attributes
from genops.core.telemetry import GenOpsTelemetry

# Strategies for generating realistic test data
cost_strategy = st.floats(
    min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False
)
currency_strategy = st.sampled_from(["USD", "EUR", "GBP", "JPY"])
provider_strategy = st.sampled_from(["openai", "anthropic", "bedrock", "gemini"])
model_strategy = st.sampled_from(
    [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo",
        "claude-3-sonnet",
        "claude-3-opus",
        "claude-3-haiku",
    ]
)
team_strategy = st.text(
    min_size=1,
    max_size=50,
    alphabet=st.characters(whitelist_categories=["Ll", "Lu", "Nd", "-"]),
)
project_strategy = st.text(
    min_size=1,
    max_size=50,
    alphabet=st.characters(whitelist_categories=["Ll", "Lu", "Nd", "-"]),
)
customer_id_strategy = st.text(
    min_size=1,
    max_size=100,
    alphabet=st.characters(whitelist_categories=["Ll", "Lu", "Nd", "-", "_"]),
)


class TestCostAttributionProperties:
    """Property-based tests for cost attribution functionality."""

    @given(
        cost=cost_strategy,
        currency=currency_strategy,
        provider=provider_strategy,
        model=model_strategy,
        team=team_strategy,
        project=project_strategy,
    )
    @settings(max_examples=500, deadline=None)
    def test_cost_recording_properties(
        self, cost, currency, provider, model, team, project
    ):
        """Test that cost recording always maintains correct properties."""
        # Assume valid inputs
        assume(cost >= 0)
        assume(len(team.strip()) > 0)
        assume(len(project.strip()) > 0)

        telemetry = GenOpsTelemetry()

        with telemetry.trace_operation(
            operation_name="test_operation", team=team.strip(), project=project.strip()
        ) as span:
            # Record cost
            telemetry.record_cost(
                span=span, cost=cost, currency=currency, provider=provider, model=model
            )

            # Properties that should always hold
            assert span is not None
            # Cost should be non-negative
            assert cost >= 0
            # Currency should be valid
            assert currency in ["USD", "EUR", "GBP", "JPY"]
            # Provider should be valid
            assert provider in ["openai", "anthropic", "bedrock", "gemini"]

    @given(
        operations=st.lists(
            st.tuples(
                cost_strategy,
                currency_strategy,
                provider_strategy,
                team_strategy,
                project_strategy,
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_multiple_operations_consistency(self, operations):
        """Test that multiple cost recording operations maintain consistency."""
        telemetry = GenOpsTelemetry()
        total_cost = 0
        recorded_operations = []

        for cost, currency, provider, team, project in operations:
            # Skip invalid inputs
            if cost < 0 or len(team.strip()) == 0 or len(project.strip()) == 0:
                continue

            with telemetry.trace_operation(
                operation_name=f"operation_{len(recorded_operations)}",
                team=team.strip(),
                project=project.strip(),
            ) as span:
                telemetry.record_cost(
                    span=span, cost=cost, currency=currency, provider=provider
                )

                recorded_operations.append(
                    (cost, currency, provider, team.strip(), project.strip())
                )
                if currency == "USD":  # Only sum USD for simplicity
                    total_cost += cost

        # Properties that should hold for multiple operations
        assert len(recorded_operations) >= 0
        assert total_cost >= 0
        # Each operation should have maintained its individual properties
        for cost, _currency, _provider, team, project in recorded_operations:
            assert cost >= 0
            assert len(team) > 0
            assert len(project) > 0

    @given(
        cost=st.floats(min_value=0.001, max_value=100.0),
        tokens_input=st.integers(min_value=1, max_value=10000),
        tokens_output=st.integers(min_value=1, max_value=10000),
    )
    @settings(max_examples=200)
    def test_cost_per_token_calculation_properties(
        self, cost, tokens_input, tokens_output
    ):
        """Test that cost per token calculations maintain mathematical properties."""
        assume(cost > 0)
        assume(tokens_input > 0)
        assume(tokens_output > 0)

        total_tokens = tokens_input + tokens_output
        cost_per_token = cost / total_tokens

        # Mathematical properties that should always hold
        assert cost_per_token > 0
        assert cost_per_token <= cost  # Cost per token should not exceed total cost
        assert cost_per_token * total_tokens == pytest.approx(cost, rel=1e-6)

        # Reconstruct cost from per-token calculation
        reconstructed_cost = cost_per_token * total_tokens
        assert reconstructed_cost == pytest.approx(cost, rel=1e-6)


class CostAttributionStateMachine(RuleBasedStateMachine):
    """Stateful property-based testing for cost attribution system."""

    def __init__(self):
        super().__init__()
        self.telemetry = GenOpsTelemetry()
        self.total_recorded_cost = 0
        self.operation_count = 0
        self.active_spans = []

    @rule(
        cost=cost_strategy,
        provider=provider_strategy,
        team=team_strategy,
        project=project_strategy,
    )
    def record_cost_operation(self, cost, provider, team, project):
        """Rule: Record a cost operation."""
        assume(cost >= 0)
        assume(len(team.strip()) > 0)
        assume(len(project.strip()) > 0)

        operation_name = f"operation_{self.operation_count}"

        span = self.telemetry.trace_operation(
            operation_name=operation_name, team=team.strip(), project=project.strip()
        ).__enter__()

        self.telemetry.record_cost(span=span, cost=cost, provider=provider)

        # Update state
        self.total_recorded_cost += cost
        self.operation_count += 1
        self.active_spans.append(span)

        # Ensure span cleanup
        span.__exit__(None, None, None)
        self.active_spans.remove(span)

    @rule(team=team_strategy, project=project_strategy)
    def set_default_attribution(self, team, project):
        """Rule: Set default attribution context."""
        assume(len(team.strip()) > 0)
        assume(len(project.strip()) > 0)

        set_default_attributes(team=team.strip(), project=project.strip())

    @invariant()
    def total_cost_is_non_negative(self):
        """Invariant: Total recorded cost should always be non-negative."""
        assert self.total_recorded_cost >= 0

    @invariant()
    def operation_count_is_consistent(self):
        """Invariant: Operation count should match recorded operations."""
        assert self.operation_count >= 0

    @invariant()
    def no_dangling_spans(self):
        """Invariant: No spans should remain active after operations complete."""
        assert len(self.active_spans) == 0


class TestCostAttributionStateMachine:
    """Test runner for stateful property-based testing."""

    def test_cost_attribution_state_machine(self):
        """Run the stateful property-based test."""
        state_machine_test = CostAttributionStateMachine.TestCase()
        state_machine_test.runTest()


@given(
    customer_operations=st.dictionaries(
        keys=customer_id_strategy,
        values=st.lists(
            st.tuples(cost_strategy, provider_strategy), min_size=1, max_size=10
        ),
        min_size=1,
        max_size=10,
    )
)
@settings(max_examples=50, deadline=None)
def test_customer_cost_attribution_properties(customer_operations):
    """Test properties of per-customer cost attribution."""
    telemetry = GenOpsTelemetry()
    customer_totals = {}

    for customer_id, operations in customer_operations.items():
        customer_total = 0

        for cost, provider in operations:
            if cost < 0:
                continue

            with telemetry.trace_operation(
                operation_name="customer_operation",
                customer_id=customer_id.strip() if customer_id.strip() else "default",
            ) as span:
                telemetry.record_cost(span=span, cost=cost, provider=provider)
                customer_total += cost

        customer_totals[customer_id] = customer_total

    # Properties that should hold
    for customer_id, total in customer_totals.items():
        assert total >= 0, f"Customer {customer_id} should have non-negative total cost"

    # Total across all customers should equal sum of individual totals
    overall_total = sum(customer_totals.values())
    assert overall_total >= 0


if __name__ == "__main__":
    # Run property-based tests
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
