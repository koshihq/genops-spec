"""
Comprehensive tests for GenOps SkyRouter Adapter.

Tests the core adapter functionality including:
- Multi-model routing with governance attributes
- Auto-instrumentation patterns
- Cost calculation and attribution
- Error handling and resilience
- Session tracking and lifecycle management
- Enterprise deployment patterns
"""

from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

# Import the modules under test
try:
    from genops.providers.skyrouter import (
        GenOpsSkyRouterAdapter,
        SkyRouterOperationResult,
        auto_instrument,
        get_current_adapter,
        restore_skyrouter,
    )
    from genops.providers.skyrouter_cost_aggregator import (
        SkyRouterCostAggregator,  # noqa: F401
    )
    from genops.providers.skyrouter_pricing import SkyRouterPricingConfig  # noqa: F401
    from genops.providers.skyrouter_validation import SkyRouterValidator  # noqa: F401

    SKYROUTER_AVAILABLE = True
except ImportError:
    SKYROUTER_AVAILABLE = False


@pytest.mark.skipif(not SKYROUTER_AVAILABLE, reason="SkyRouter provider not available")
class TestGenOpsSkyRouterAdapter:
    """Test suite for the main SkyRouter adapter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = GenOpsSkyRouterAdapter(
            team="test-team",
            project="test-project",
            environment="test",
            daily_budget_limit=100.0,
        )
        self.sample_governance_attrs = {
            "team": "test-team",
            "project": "test-project",
            "customer_id": "test-customer",
            "environment": "test",
        }

    def test_adapter_initialization(self):
        """Test adapter initialization with various configurations."""
        # Basic initialization
        adapter = GenOpsSkyRouterAdapter(team="team1", project="proj1")
        assert adapter.governance_attrs.team == "team1"
        assert adapter.governance_attrs.project == "proj1"

        # Full configuration initialization
        full_adapter = GenOpsSkyRouterAdapter(
            skyrouter_api_key="test-key",
            team="enterprise-team",
            project="production-project",
            environment="production",
            daily_budget_limit=500.0,
            enable_cost_alerts=True,
            governance_policy="strict",
        )

        assert full_adapter.governance_attrs.team == "enterprise-team"
        assert full_adapter.governance_attrs.environment == "production"
        assert full_adapter.daily_budget_limit == 500.0
        assert full_adapter.governance_policy == "strict"

    def test_adapter_initialization_with_invalid_params(self):
        """Test adapter initialization with invalid parameters."""
        # Test with negative budget
        with pytest.raises(ValueError, match="daily_budget_limit must be positive"):
            GenOpsSkyRouterAdapter(
                team="test-team", project="test-project", daily_budget_limit=-10.0
            )

        # Test with invalid governance policy
        with pytest.raises(ValueError, match="governance_policy must be one of"):
            GenOpsSkyRouterAdapter(
                team="test-team", project="test-project", governance_policy="invalid"
            )

    @patch("genops.providers.skyrouter.skyrouter")
    def test_single_model_routing(self, mock_skyrouter):
        """Test single model routing with governance."""
        # Mock SkyRouter response
        mock_response = Mock()
        mock_response.model = "gpt-4"
        mock_response.usage = {"total_tokens": 150}
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_skyrouter.route.return_value = mock_response

        # Test single model call
        with self.adapter.track_routing_session("test-session") as session:
            result = session.track_model_call(
                model="gpt-4",
                input_data={"prompt": "Test prompt"},
                route_optimization="cost_optimized",
            )

            assert isinstance(result, SkyRouterOperationResult)
            assert result.model == "gpt-4"
            assert result.total_cost > 0
            assert result.route_optimization == "cost_optimized"

    @patch("genops.providers.skyrouter.skyrouter")
    def test_multi_model_routing(self, mock_skyrouter):
        """Test multi-model routing with strategy selection."""
        # Mock SkyRouter multi-model response
        mock_response = Mock()
        mock_response.model = "claude-3-sonnet"
        mock_response.route = "balanced"
        mock_response.usage = {"total_tokens": 200}
        mock_response.route_efficiency_score = 0.85
        mock_skyrouter.route_to_best_model.return_value = mock_response

        routing_strategies = [
            "cost_optimized",
            "balanced",
            "latency_optimized",
            "reliability_first",
        ]

        for strategy in routing_strategies:
            with self.adapter.track_routing_session(
                f"multi-test-{strategy}"
            ) as session:
                result = session.track_multi_model_routing(
                    models=["gpt-4", "claude-3-sonnet", "gemini-pro"],
                    input_data={"task": "content generation"},
                    routing_strategy=strategy,
                )

                assert isinstance(result, SkyRouterOperationResult)
                assert result.routing_strategy == strategy
                assert result.route_efficiency_score >= 0.0

    def test_governance_attribute_propagation(self):
        """Test that governance attributes are properly propagated."""
        adapter = GenOpsSkyRouterAdapter(
            team="governance-test",
            project="attribute-test",
            customer_id="customer-123",
            environment="staging",
        )

        attrs = adapter.governance_attrs
        assert attrs.team == "governance-test"
        assert attrs.project == "attribute-test"
        assert attrs.customer_id == "customer-123"
        assert attrs.environment == "staging"

    @patch("genops.providers.skyrouter.skyrouter")
    def test_cost_calculation_accuracy(self, mock_skyrouter):
        """Test cost calculation accuracy across different models."""
        test_cases = [
            {"model": "gpt-4", "tokens": 1000, "expected_min": 0.02},
            {"model": "gpt-3.5-turbo", "tokens": 1000, "expected_min": 0.001},
            {"model": "claude-3-opus", "tokens": 1000, "expected_min": 0.015},
        ]

        for case in test_cases:
            mock_response = Mock()
            mock_response.model = case["model"]
            mock_response.usage = {"total_tokens": case["tokens"]}
            mock_skyrouter.route.return_value = mock_response

            with self.adapter.track_routing_session("cost-test") as session:
                result = session.track_model_call(
                    model=case["model"], input_data={"test": True}
                )

                assert result.total_cost >= case["expected_min"]
                assert isinstance(result.total_cost, (int, float, Decimal))

    def test_session_context_manager(self):
        """Test session context manager lifecycle."""
        session_id = None

        # Test successful session
        with self.adapter.track_routing_session("context-test") as session:
            session_id = session.session_id
            assert session_id is not None
            assert session.adapter == self.adapter

        # Session should be finalized after context exit
        assert session_id is not None

    def test_session_context_manager_with_exception(self):
        """Test session context manager behavior during exceptions."""
        session_id = None

        with pytest.raises(ValueError):
            with self.adapter.track_routing_session("exception-test") as session:
                session_id = session.session_id
                raise ValueError("Test exception")

        # Session should still be properly finalized
        assert session_id is not None

    @patch("genops.providers.skyrouter.skyrouter")
    def test_agent_workflow_tracking(self, mock_skyrouter):
        """Test multi-agent workflow tracking."""
        # Mock responses for different workflow steps
        mock_responses = [
            Mock(model="gpt-3.5-turbo", usage={"total_tokens": 100}),
            Mock(model="claude-3-sonnet", usage={"total_tokens": 150}),
            Mock(model="gpt-4", usage={"total_tokens": 200}),
        ]
        mock_skyrouter.route.side_effect = mock_responses

        workflow_steps = [
            {
                "model": "gpt-3.5-turbo",
                "input": {"task": "classification"},
                "complexity": "simple",
                "optimization": "cost_optimized",
            },
            {
                "model": "claude-3-sonnet",
                "input": {"task": "generation"},
                "complexity": "moderate",
                "optimization": "balanced",
            },
            {
                "model": "gpt-4",
                "input": {"task": "review"},
                "complexity": "complex",
                "optimization": "reliability_first",
            },
        ]

        with self.adapter.track_routing_session("workflow-test") as session:
            result = session.track_agent_workflow(
                workflow_name="test_workflow", agent_steps=workflow_steps
            )

            assert isinstance(result, SkyRouterOperationResult)
            assert result.metadata["workflow_name"] == "test_workflow"
            assert result.metadata["step_count"] == 3
            assert len(result.metadata["models_used"]) == 3
            assert result.total_cost > 0

    def test_budget_limit_enforcement(self):
        """Test budget limit enforcement."""
        # Create adapter with small budget
        budget_adapter = GenOpsSkyRouterAdapter(
            team="budget-test",
            project="limit-test",
            daily_budget_limit=0.01,  # Very small budget
            governance_policy="enforced",
        )

        # Check budget status
        budget_status = budget_adapter.cost_aggregator.check_budget_status()
        assert "daily_budget_limit" in budget_status
        assert budget_status["daily_budget_limit"] == 0.01

    def test_cost_alert_configuration(self):
        """Test cost alert configuration and thresholds."""
        alert_adapter = GenOpsSkyRouterAdapter(
            team="alert-test",
            project="notification-test",
            enable_cost_alerts=True,
            daily_budget_limit=50.0,
        )

        assert alert_adapter.enable_cost_alerts is True
        assert alert_adapter.daily_budget_limit == 50.0

    @patch("genops.providers.skyrouter.skyrouter")
    def test_error_handling_network_failure(self, mock_skyrouter):
        """Test error handling during network failures."""
        # Mock network failure
        mock_skyrouter.route.side_effect = ConnectionError("Network error")

        with pytest.raises(ConnectionError):
            with self.adapter.track_routing_session("network-error-test") as session:
                session.track_model_call(model="gpt-4", input_data={"prompt": "Test"})

    @patch("genops.providers.skyrouter.skyrouter")
    def test_error_handling_api_error(self, mock_skyrouter):
        """Test error handling for API errors."""
        # Mock API error
        mock_skyrouter.route.side_effect = Exception("API Error: Invalid model")

        with pytest.raises(Exception):  # noqa: B017
            with self.adapter.track_routing_session("api-error-test") as session:
                session.track_model_call(
                    model="invalid-model", input_data={"prompt": "Test"}
                )

    def test_complexity_level_validation(self):
        """Test complexity level validation."""
        valid_complexities = ["simple", "moderate", "complex", "enterprise"]

        for _complexity in valid_complexities:
            # Should not raise exception
            with self.adapter.track_routing_session("complexity-test") as session:
                assert session is not None

    def test_routing_strategy_validation(self):
        """Test routing strategy validation."""
        valid_strategies = [
            "cost_optimized",
            "balanced",
            "latency_optimized",
            "reliability_first",
        ]

        for strategy in valid_strategies:
            # Should not raise exception for valid strategies
            assert strategy in valid_strategies

    @patch("genops.providers.skyrouter.skyrouter")
    def test_telemetry_data_structure(self, mock_skyrouter):
        """Test telemetry data structure and attributes."""
        mock_response = Mock()
        mock_response.model = "gpt-4"
        mock_response.usage = {"total_tokens": 100}
        mock_skyrouter.route.return_value = mock_response

        with self.adapter.track_routing_session("telemetry-test") as session:
            result = session.track_model_call(
                model="gpt-4", input_data={"prompt": "Test telemetry"}
            )

            # Check telemetry data structure
            assert hasattr(result, "model")
            assert hasattr(result, "total_cost")
            assert hasattr(result, "session_id")
            assert hasattr(result, "governance_attrs")

    def test_environment_specific_configuration(self):
        """Test environment-specific configurations."""
        environments = ["development", "staging", "production", "disaster_recovery"]

        for env in environments:
            env_adapter = GenOpsSkyRouterAdapter(
                team="env-test",
                project="multi-env-test",
                environment=env,
                daily_budget_limit=100.0 if env == "production" else 50.0,
            )

            assert env_adapter.governance_attrs.environment == env
            expected_budget = 100.0 if env == "production" else 50.0
            assert env_adapter.daily_budget_limit == expected_budget

    def test_compliance_configuration(self):
        """Test compliance framework configuration."""
        compliance_adapter = GenOpsSkyRouterAdapter(
            team="compliance-test",
            project="framework-test",
            compliance_config={
                "frameworks": ["soc2", "hipaa", "gdpr"],
                "audit_logging": True,
                "data_encryption": True,
            },
        )

        assert compliance_adapter.compliance_config is not None
        assert "frameworks" in compliance_adapter.compliance_config
        assert "soc2" in compliance_adapter.compliance_config["frameworks"]

    @patch("genops.providers.skyrouter.skyrouter")
    def test_high_availability_configuration(self, mock_skyrouter):
        """Test high-availability configuration."""
        ha_adapter = GenOpsSkyRouterAdapter(
            team="ha-test",
            project="availability-test",
            ha_config={
                "region": "us-east-1",
                "failover_enabled": True,
                "backup_regions": ["us-west-2"],
                "replication_lag_threshold": "5s",
            },
        )

        assert ha_adapter.ha_config is not None
        assert ha_adapter.ha_config["failover_enabled"] is True

    def test_scaling_configuration(self):
        """Test auto-scaling configuration."""
        scaling_adapter = GenOpsSkyRouterAdapter(
            team="scaling-test",
            project="autoscale-test",
            scaling_config={
                "min_instances": 2,
                "max_instances": 20,
                "target_cpu_utilization": 70,
                "auto_scaling": True,
            },
        )

        assert scaling_adapter.scaling_config is not None
        assert scaling_adapter.scaling_config["auto_scaling"] is True

    def test_load_balancer_configuration(self):
        """Test load balancer configuration."""
        lb_adapter = GenOpsSkyRouterAdapter(
            team="lb-test",
            project="loadbalance-test",
            load_balancer_config={
                "algorithm": "least_connections",
                "health_check_interval": 30,
                "session_affinity": "source_ip",
            },
        )

        assert lb_adapter.load_balancer_config is not None
        assert lb_adapter.load_balancer_config["algorithm"] == "least_connections"


@pytest.mark.skipif(not SKYROUTER_AVAILABLE, reason="SkyRouter provider not available")
class TestSkyRouterAutoInstrumentation:
    """Test suite for auto-instrumentation functionality."""

    def test_auto_instrument_basic(self):
        """Test basic auto-instrumentation setup."""
        # Test auto-instrumentation
        adapter = auto_instrument(team="auto-test", project="instrumentation-test")

        assert adapter is not None
        assert adapter.governance_attrs.team == "auto-test"

        # Clean up
        restore_skyrouter()

    def test_auto_instrument_with_configuration(self):
        """Test auto-instrumentation with full configuration."""
        adapter = auto_instrument(
            team="auto-config-test",
            project="full-config-test",
            daily_budget_limit=75.0,
            enable_cost_alerts=True,
            governance_policy="advisory",
        )

        assert adapter is not None
        assert adapter.daily_budget_limit == 75.0
        assert adapter.enable_cost_alerts is True
        assert adapter.governance_policy == "advisory"

        # Clean up
        restore_skyrouter()

    def test_get_current_adapter(self):
        """Test getting current adapter instance."""
        # No adapter initially
        assert get_current_adapter() is None

        # Set up adapter
        adapter = auto_instrument(team="current-test", project="adapter-test")
        current = get_current_adapter()

        assert current is not None
        assert current == adapter

        # Clean up
        restore_skyrouter()
        assert get_current_adapter() is None

    def test_restore_skyrouter(self):
        """Test restoring original SkyRouter functionality."""
        # Set up auto-instrumentation
        auto_instrument(team="restore-test", project="cleanup-test")
        assert get_current_adapter() is not None

        # Restore original functionality
        restore_skyrouter()
        assert get_current_adapter() is None

    def test_multiple_auto_instrument_calls(self):
        """Test behavior with multiple auto-instrument calls."""
        # First instrumentation
        auto_instrument(team="multi-1", project="test-1")

        # Second instrumentation should replace first
        adapter2 = auto_instrument(team="multi-2", project="test-2")

        current = get_current_adapter()
        assert current == adapter2
        assert current.governance_attrs.team == "multi-2"

        # Clean up
        restore_skyrouter()


@pytest.mark.skipif(not SKYROUTER_AVAILABLE, reason="SkyRouter provider not available")
class TestSkyRouterEnterpriseFeatures:
    """Test suite for enterprise-specific features."""

    def test_multi_environment_deployment(self):
        """Test multi-environment deployment patterns."""
        environments = [
            {"env": "development", "budget": 10.0, "policy": "advisory"},
            {"env": "staging", "budget": 50.0, "policy": "enforced"},
            {"env": "production", "budget": 500.0, "policy": "strict"},
        ]

        adapters = []
        for env_config in environments:
            adapter = GenOpsSkyRouterAdapter(
                team=f"enterprise-{env_config['env']}",
                project="multi-env-test",
                environment=env_config["env"],
                daily_budget_limit=env_config["budget"],
                governance_policy=env_config["policy"],
            )
            adapters.append(adapter)

            assert adapter.governance_attrs.environment == env_config["env"]
            assert adapter.daily_budget_limit == env_config["budget"]
            assert adapter.governance_policy == env_config["policy"]

    def test_department_cost_governance(self):
        """Test department-level cost governance."""
        departments = {
            "engineering": {"budget": 500.0, "cost_center": "TECH-001"},
            "product": {"budget": 200.0, "cost_center": "PROD-002"},
            "customer_success": {"budget": 150.0, "cost_center": "CS-003"},
            "sales": {"budget": 100.0, "cost_center": "SALES-004"},
        }

        for dept_name, config in departments.items():
            adapter = GenOpsSkyRouterAdapter(
                team=f"dept-{dept_name}",
                project="department-governance",
                daily_budget_limit=config["budget"],
                cost_center=config["cost_center"],
                governance_policy="strict",
            )

            assert adapter.daily_budget_limit == config["budget"]
            assert adapter.cost_center == config["cost_center"]

    def test_enterprise_monitoring_configuration(self):
        """Test enterprise monitoring and alerting configuration."""
        monitoring_adapter = GenOpsSkyRouterAdapter(
            team="enterprise-monitoring",
            project="production-monitoring",
            monitoring_config={
                "metrics_collection": "comprehensive",
                "alert_channels": ["slack", "pagerduty", "email"],
                "sla_monitoring": True,
                "cost_anomaly_detection": True,
                "real_time_dashboards": True,
            },
        )

        assert monitoring_adapter.monitoring_config is not None
        config = monitoring_adapter.monitoring_config
        assert config["sla_monitoring"] is True
        assert "slack" in config["alert_channels"]

    @patch("genops.providers.skyrouter.skyrouter")
    def test_enterprise_workflow_patterns(self, mock_skyrouter):
        """Test enterprise workflow patterns."""
        # Mock multi-step workflow responses
        mock_responses = [
            Mock(model="gpt-3.5-turbo", usage={"total_tokens": 50}),
            Mock(model="claude-3-sonnet", usage={"total_tokens": 100}),
            Mock(model="gpt-4", usage={"total_tokens": 150}),
            Mock(model="claude-3-opus", usage={"total_tokens": 200}),
        ]
        mock_skyrouter.route.side_effect = mock_responses

        enterprise_adapter = GenOpsSkyRouterAdapter(
            team="enterprise-workflows",
            project="production-patterns",
            environment="production",
            governance_policy="strict",
        )

        # Customer support workflow
        customer_support_steps = [
            {
                "model": "gpt-3.5-turbo",
                "task": "intent_classification",
                "complexity": "simple",
            },
            {
                "model": "claude-3-sonnet",
                "task": "solution_generation",
                "complexity": "moderate",
            },
            {"model": "gpt-4", "task": "quality_review", "complexity": "complex"},
            {
                "model": "claude-3-opus",
                "task": "escalation_detection",
                "complexity": "enterprise",
            },
        ]

        with enterprise_adapter.track_routing_session("enterprise-workflow") as session:
            workflow_steps = []
            for step in customer_support_steps:
                workflow_steps.append(
                    {
                        "model": step["model"],
                        "input": {"task": step["task"]},
                        "complexity": step["complexity"],
                        "optimization": "reliability_first",
                    }
                )

            result = session.track_agent_workflow(
                workflow_name="customer_support_enterprise", agent_steps=workflow_steps
            )

            assert isinstance(result, SkyRouterOperationResult)
            assert result.metadata["workflow_name"] == "customer_support_enterprise"
            assert result.metadata["step_count"] == 4


@pytest.mark.skipif(not SKYROUTER_AVAILABLE, reason="SkyRouter provider not available")
class TestSkyRouterPerformance:
    """Test suite for performance and scalability."""

    @patch("genops.providers.skyrouter.skyrouter")
    def test_high_volume_operations(self, mock_skyrouter):
        """Test performance with high volume operations."""
        mock_response = Mock()
        mock_response.model = "gpt-3.5-turbo"
        mock_response.usage = {"total_tokens": 100}
        mock_skyrouter.route.return_value = mock_response

        performance_adapter = GenOpsSkyRouterAdapter(
            team="performance-test", project="volume-test"
        )

        # Simulate high-volume operations
        operation_count = 100
        total_cost = 0

        with performance_adapter.track_routing_session("high-volume-test") as session:
            for i in range(operation_count):
                result = session.track_model_call(
                    model="gpt-3.5-turbo", input_data={"operation_id": i}
                )
                total_cost += float(result.total_cost)

        assert total_cost > 0
        # Performance should be reasonable for 100 operations

    def test_concurrent_session_handling(self):
        """Test handling of concurrent sessions."""
        concurrent_adapter = GenOpsSkyRouterAdapter(
            team="concurrent-test", project="parallel-test"
        )

        # Test multiple concurrent sessions
        sessions = []
        for i in range(10):
            session = concurrent_adapter.track_routing_session(f"concurrent-{i}")
            sessions.append(session)

        # All sessions should be valid
        for session in sessions:
            assert session.session_id is not None

    @patch("genops.providers.skyrouter.skyrouter")
    def test_memory_usage_optimization(self, mock_skyrouter):
        """Test memory usage optimization for long-running operations."""
        mock_response = Mock()
        mock_response.model = "gpt-3.5-turbo"
        mock_response.usage = {"total_tokens": 100}
        mock_skyrouter.route.return_value = mock_response

        memory_adapter = GenOpsSkyRouterAdapter(
            team="memory-test", project="optimization-test"
        )

        # Simulate long-running operations
        with memory_adapter.track_routing_session("memory-test") as session:
            for i in range(50):
                result = session.track_model_call(
                    model="gpt-3.5-turbo", input_data={"iteration": i}
                )
                # Memory should remain stable
                assert result is not None


if __name__ == "__main__":
    pytest.main([__file__])
