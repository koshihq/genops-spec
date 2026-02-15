"""Performance benchmarks for GenOps AI using pytest-benchmark."""

import pytest

from genops.core.context import set_context
from genops.core.policy import PolicyConfig, PolicyEngine, PolicyResult
from genops.core.telemetry import GenOpsTelemetry
from genops.core.validation import validate_tags


@pytest.fixture
def policy_engine():
    """Create a policy engine with sample policies."""
    engine = PolicyEngine()
    engine.register_policy(
        PolicyConfig(
            name="cost_limit",
            enforcement_level=PolicyResult.BLOCKED,
            conditions={"max_cost": 10.0},
        )
    )
    engine.register_policy(
        PolicyConfig(
            name="rate_limit",
            enforcement_level=PolicyResult.RATE_LIMITED,
            conditions={"max_requests": 1000, "time_window": 3600},
        )
    )
    return engine


@pytest.fixture
def telemetry():
    """Create a telemetry instance."""
    return GenOpsTelemetry("benchmark-test")


@pytest.mark.benchmark
def test_context_creation(benchmark):
    """Benchmark context setup."""

    def create_context():
        return set_context(
            team="benchmark-team",
            project="benchmark-project",
            environment="production",
        )

    benchmark(create_context)


@pytest.mark.benchmark
def test_policy_evaluation_allowed(benchmark, policy_engine):
    """Benchmark policy evaluation for allowed operations."""
    context = {"cost": 5.0}

    def evaluate():
        return policy_engine.evaluate_policy("cost_limit", context)

    benchmark(evaluate)


@pytest.mark.benchmark
def test_policy_evaluation_blocked(benchmark, policy_engine):
    """Benchmark policy evaluation for blocked operations."""
    context = {"cost": 15.0}

    def evaluate():
        return policy_engine.evaluate_policy("cost_limit", context)

    benchmark(evaluate)


@pytest.mark.benchmark
def test_attribute_validation(benchmark):
    """Benchmark attribute validation."""
    attrs = {
        "team": "benchmark-team",
        "project": "benchmark-project",
        "model": "gpt-4",
        "provider": "openai",
        "environment": "production",
        "customer_id": "cust-123",
    }

    benchmark(validate_tags, attrs)


@pytest.mark.benchmark
def test_telemetry_trace_operation(benchmark, telemetry):
    """Benchmark telemetry trace operation overhead."""

    def trace_op():
        with telemetry.trace_operation("benchmark.operation"):
            pass

    benchmark(trace_op)


@pytest.mark.benchmark
def test_multiple_policy_evaluation(benchmark, policy_engine):
    """Benchmark evaluating multiple policies sequentially."""
    context = {"cost": 5.0, "request_count": 50, "time_window": 3600}

    def evaluate_all():
        policy_engine.evaluate_policy("cost_limit", context)
        policy_engine.evaluate_policy("rate_limit", context)

    benchmark(evaluate_all)
