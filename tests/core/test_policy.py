"""Tests for GenOps AI policy engine."""

from unittest.mock import MagicMock, patch

import pytest

from genops.core.policy import (
    PolicyConfig,
    PolicyEngine,
    PolicyResult,
    PolicyViolationError,
    enforce_policy,
    register_policy,
)


class TestPolicyConfig:
    """Test PolicyConfig class."""

    def test_policy_config_creation(self):
        """Test basic policy configuration creation."""
        policy = PolicyConfig(
            name="test_policy",
            description="Test policy description",
            enforcement_level=PolicyResult.BLOCKED,
            conditions={"max_cost": 5.0},
        )

        assert policy.name == "test_policy"
        assert policy.description == "Test policy description"
        assert policy.enabled is True  # Default value
        assert policy.enforcement_level == PolicyResult.BLOCKED
        assert policy.conditions["max_cost"] == 5.0

    def test_policy_config_defaults(self):
        """Test policy configuration with default values."""
        policy = PolicyConfig(name="minimal_policy")

        assert policy.name == "minimal_policy"
        assert policy.description == ""
        assert policy.enabled is True
        assert policy.enforcement_level == PolicyResult.BLOCKED
        assert policy.conditions == {}

    def test_policy_config_disabled(self):
        """Test disabled policy configuration."""
        policy = PolicyConfig(
            name="disabled_policy",
            enabled=False,
            enforcement_level=PolicyResult.WARNING,
        )

        assert policy.enabled is False
        assert policy.enforcement_level == PolicyResult.WARNING


class TestPolicyEngine:
    """Test PolicyEngine class."""

    def test_policy_engine_initialization(self):
        """Test PolicyEngine initialization."""
        engine = PolicyEngine()
        assert len(engine.policies) == 0

    def test_register_policy(self):
        """Test policy registration."""
        engine = PolicyEngine()
        policy = PolicyConfig(name="cost_limit", conditions={"max_cost": 10.0})

        engine.register_policy(policy)
        assert len(engine.policies) == 1
        assert "cost_limit" in engine.policies
        assert engine.policies["cost_limit"] == policy

    def test_register_duplicate_policy_overwrites(self):
        """Test that registering a duplicate policy overwrites the existing one."""
        engine = PolicyEngine()

        policy1 = PolicyConfig(name="test_policy", conditions={"max_cost": 5.0})
        policy2 = PolicyConfig(name="test_policy", conditions={"max_cost": 10.0})

        engine.register_policy(policy1)
        assert engine.policies["test_policy"].conditions["max_cost"] == 5.0

        engine.register_policy(policy2)
        assert engine.policies["test_policy"].conditions["max_cost"] == 10.0

    def test_evaluate_policy_disabled(self):
        """Test that disabled policies return ALLOWED."""
        engine = PolicyEngine()
        policy = PolicyConfig(
            name="disabled_policy",
            enabled=False,
            enforcement_level=PolicyResult.BLOCKED,
        )
        engine.register_policy(policy)

        result = engine.evaluate_policy("disabled_policy", {})
        assert result.result == PolicyResult.ALLOWED
        assert "disabled" in result.reason.lower()

    def test_evaluate_policy_not_found(self):
        """Test evaluating non-existent policy."""
        engine = PolicyEngine()

        result = engine.evaluate_policy("non_existent", {})
        assert result.result == PolicyResult.ALLOWED
        assert "not found" in result.reason.lower()

    def test_cost_limit_policy_allowed(self):
        """Test cost limit policy allows requests under limit."""
        engine = PolicyEngine()
        policy = PolicyConfig(
            name="cost_limit",
            enforcement_level=PolicyResult.BLOCKED,
            conditions={"max_cost": 5.0},
        )
        engine.register_policy(policy)

        context = {"cost": 3.0}
        result = engine.evaluate_policy("cost_limit", context)

        assert result.result == PolicyResult.ALLOWED
        assert result.policy_name == "cost_limit"

    def test_cost_limit_policy_blocked(self):
        """Test cost limit policy blocks requests over limit."""
        engine = PolicyEngine()
        policy = PolicyConfig(
            name="cost_limit",
            enforcement_level=PolicyResult.BLOCKED,
            conditions={"max_cost": 5.0},
        )
        engine.register_policy(policy)

        context = {"cost": 7.0}
        result = engine.evaluate_policy("cost_limit", context)

        assert result.result == PolicyResult.BLOCKED
        assert "cost limit exceeded" in result.reason.lower()
        assert result.metadata["limit"] == 5.0
        assert result.metadata["actual"] == 7.0

    def test_rate_limit_policy_allowed(self):
        """Test rate limit policy allows requests within limit."""
        engine = PolicyEngine()
        policy = PolicyConfig(
            name="rate_limit",
            enforcement_level=PolicyResult.RATE_LIMITED,
            conditions={"max_requests": 100, "time_window": 3600},
        )
        engine.register_policy(policy)

        context = {"request_count": 50, "time_window": 3600}
        result = engine.evaluate_policy("rate_limit", context)

        assert result.result == PolicyResult.ALLOWED

    def test_rate_limit_policy_exceeded(self):
        """Test rate limit policy blocks requests over limit."""
        engine = PolicyEngine()
        policy = PolicyConfig(
            name="rate_limit",
            enforcement_level=PolicyResult.RATE_LIMITED,
            conditions={"max_requests": 100, "time_window": 3600},
        )
        engine.register_policy(policy)

        context = {"request_count": 150, "time_window": 3600}
        result = engine.evaluate_policy("rate_limit", context)

        assert result.result == PolicyResult.RATE_LIMITED
        assert "rate limit exceeded" in result.reason.lower()

    def test_content_filter_policy_allowed(self):
        """Test content filter allows safe content."""
        engine = PolicyEngine()
        policy = PolicyConfig(
            name="content_filter",
            enforcement_level=PolicyResult.BLOCKED,
            conditions={"blocked_patterns": ["violence", "explicit"]},
        )
        engine.register_policy(policy)

        context = {"content": "This is a safe, educational message about AI."}
        result = engine.evaluate_policy("content_filter", context)

        assert result.result == PolicyResult.ALLOWED

    def test_content_filter_policy_blocked(self):
        """Test content filter blocks unsafe content."""
        engine = PolicyEngine()
        policy = PolicyConfig(
            name="content_filter",
            enforcement_level=PolicyResult.BLOCKED,
            conditions={"blocked_patterns": ["violence", "explicit"]},
        )
        engine.register_policy(policy)

        context = {"content": "This message contains violence and harmful content."}
        result = engine.evaluate_policy("content_filter", context)

        assert result.result == PolicyResult.BLOCKED
        assert "blocked content pattern" in result.reason.lower()
        assert "violence" in result.metadata["matched_patterns"]

    def test_team_access_policy_allowed(self):
        """Test team access policy allows authorized teams."""
        engine = PolicyEngine()
        policy = PolicyConfig(
            name="team_access",
            enforcement_level=PolicyResult.BLOCKED,
            conditions={"allowed_teams": ["ai-team", "platform-team"]},
        )
        engine.register_policy(policy)

        context = {"team": "ai-team"}
        result = engine.evaluate_policy("team_access", context)

        assert result.result == PolicyResult.ALLOWED

    def test_team_access_policy_blocked(self):
        """Test team access policy blocks unauthorized teams."""
        engine = PolicyEngine()
        policy = PolicyConfig(
            name="team_access",
            enforcement_level=PolicyResult.BLOCKED,
            conditions={"allowed_teams": ["ai-team", "platform-team"]},
        )
        engine.register_policy(policy)

        context = {"team": "unauthorized-team"}
        result = engine.evaluate_policy("team_access", context)

        assert result.result == PolicyResult.BLOCKED
        assert "team not authorized" in result.reason.lower()
        assert result.metadata["team"] == "unauthorized-team"

    def test_warning_enforcement_level(self):
        """Test WARNING enforcement level allows with warning."""
        engine = PolicyEngine()
        policy = PolicyConfig(
            name="warning_policy",
            enforcement_level=PolicyResult.WARNING,
            conditions={"max_cost": 5.0},
        )
        engine.register_policy(policy)

        context = {"cost": 7.0}
        result = engine.evaluate_policy("warning_policy", context)

        # Warning policies return WARNING result but don't block
        assert result.result == PolicyResult.WARNING
        assert "cost limit exceeded" in result.reason.lower()


class TestPolicyViolationError:
    """Test PolicyViolationError exception."""

    def test_policy_violation_error_creation(self):
        """Test PolicyViolationError creation."""
        metadata = {"cost": 10.0, "limit": 5.0}
        error = PolicyViolationError(
            policy_name="cost_limit", reason="Cost exceeded", metadata=metadata
        )

        assert error.policy_name == "cost_limit"
        assert error.reason == "Cost exceeded"
        assert error.metadata == metadata
        assert "Policy 'cost_limit' violation: Cost exceeded" in str(error)

    def test_policy_violation_error_no_metadata(self):
        """Test PolicyViolationError without metadata."""
        error = PolicyViolationError("test_policy", "Test violation")

        assert error.metadata == {}


class TestGlobalPolicyFunctions:
    """Test global policy registration and enforcement functions."""

    def test_register_policy_function(self):
        """Test global register_policy function."""
        # Clear any existing policies first
        from genops.core.policy import _global_policy_engine

        _global_policy_engine.policies.clear()

        register_policy(
            name="test_global_policy",
            description="Test policy",
            enforcement_level=PolicyResult.BLOCKED,
            max_cost=10.0,
        )

        assert len(_global_policy_engine.policies) == 1
        policy = _global_policy_engine.policies["test_global_policy"]
        assert policy.name == "test_global_policy"
        assert policy.conditions["max_cost"] == 10.0

    @patch("genops.core.policy._global_policy_engine")
    def test_enforce_policy_decorator_allowed(self, mock_engine):
        """Test enforce_policy decorator when policy allows operation."""
        # Mock policy evaluation to return ALLOWED
        mock_result = MagicMock()
        mock_result.result = PolicyResult.ALLOWED
        mock_engine.evaluate_policy.return_value = mock_result

        @enforce_policy(["test_policy"])
        def test_function(arg1, arg2=None):
            return f"result: {arg1}, {arg2}"

        result = test_function("hello", arg2="world")
        assert result == "result: hello, world"

        # Verify policy was evaluated
        mock_engine.evaluate_policy.assert_called_once()

    @patch("genops.core.policy._global_policy_engine")
    def test_enforce_policy_decorator_blocked(self, mock_engine):
        """Test enforce_policy decorator when policy blocks operation."""
        # Mock policy evaluation to return BLOCKED
        mock_result = MagicMock()
        mock_result.result = PolicyResult.BLOCKED
        mock_result.reason = "Test policy violation"
        mock_result.policy_name = "test_policy"
        mock_result.metadata = {}
        mock_engine.evaluate_policy.return_value = mock_result

        @enforce_policy(["test_policy"])
        def test_function():
            return "should not execute"

        with pytest.raises(PolicyViolationError) as exc_info:
            test_function()

        assert exc_info.value.policy_name == "test_policy"
        assert "Test policy violation" in str(exc_info.value)

    @patch("genops.core.policy._global_policy_engine")
    def test_enforce_policy_decorator_warning(self, mock_engine, caplog):
        """Test enforce_policy decorator with warning enforcement."""
        # Mock policy evaluation to return WARNING
        mock_result = MagicMock()
        mock_result.result = PolicyResult.WARNING
        mock_result.reason = "Cost threshold exceeded"
        mock_result.policy_name = "cost_warning"
        mock_engine.evaluate_policy.return_value = mock_result

        @enforce_policy(["cost_warning"])
        def test_function():
            return "executed with warning"

        with caplog.at_level("WARNING"):
            result = test_function()

        assert result == "executed with warning"
        assert "Policy violation warning" in caplog.text
        assert "cost_warning" in caplog.text

    @patch("genops.core.policy._global_policy_engine")
    def test_enforce_policy_multiple_policies(self, mock_engine):
        """Test enforce_policy decorator with multiple policies."""
        # Mock policy evaluations - first ALLOWED, second BLOCKED
        mock_results = [MagicMock(), MagicMock()]
        mock_results[0].result = PolicyResult.ALLOWED
        mock_results[1].result = PolicyResult.BLOCKED
        mock_results[1].reason = "Second policy blocks"
        mock_results[1].policy_name = "blocking_policy"
        mock_results[1].metadata = {}

        mock_engine.evaluate_policy.side_effect = mock_results

        @enforce_policy(["policy1", "blocking_policy"])
        def test_function():
            return "should not execute"

        with pytest.raises(PolicyViolationError):
            test_function()

        # Both policies should be evaluated
        assert mock_engine.evaluate_policy.call_count == 2

    @patch("genops.core.policy._global_policy_engine")
    def test_enforce_policy_with_telemetry(
        self, mock_engine, telemetry, mock_span_recorder
    ):
        """Test enforce_policy decorator records telemetry."""
        # Mock policy evaluation
        mock_result = MagicMock()
        mock_result.result = PolicyResult.ALLOWED
        mock_result.policy_name = "test_policy"
        mock_result.reason = "Policy allows operation"
        mock_result.metadata = {}
        mock_engine.evaluate_policy.return_value = mock_result

        @enforce_policy(["test_policy"])
        def test_function():
            # Create a span during function execution
            with telemetry.trace_operation("test.operation") as span:
                return "success"

        result = test_function()
        assert result == "success"

        # Verify span was created and policy was recorded
        spans = mock_span_recorder.get_finished_spans()
        assert len(spans) == 1
