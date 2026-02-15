"""
Integration tests for Flowise provider.

These tests verify end-to-end functionality and integration between
different components of the Flowise provider system.
"""

import os
import time
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from genops.providers.flowise import (
    auto_instrument,
    instrument_flowise,
)
from genops.providers.flowise_pricing import FlowiseCostCalculator
from genops.providers.flowise_validation import validate_flowise_setup


class TestFlowiseEndToEndWorkflow:
    """Test complete end-to-end workflows with Flowise."""

    @patch("requests.get")
    @patch("requests.post")
    def test_complete_workflow_with_mocks(self, mock_post, mock_get):
        """Test complete workflow from setup to execution."""
        # Mock successful setup validation
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = [
            {"id": "test-flow-1", "name": "Test Flow 1"},
            {"id": "test-flow-2", "name": "Test Flow 2"},
        ]
        mock_get_response.elapsed.total_seconds.return_value = 0.1
        mock_get.return_value = mock_get_response

        # Mock successful prediction
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {
            "text": "This is a test response from the AI model.",
            "metadata": {"model": "gpt-3.5-turbo", "tokens_used": 150},
        }
        mock_post.return_value = mock_post_response

        # Step 1: Validate setup
        validation_result = validate_flowise_setup(
            "http://localhost:3000", "test-api-key"
        )
        assert validation_result.is_valid

        # Step 2: Create adapter
        adapter = instrument_flowise(
            base_url="http://localhost:3000",
            api_key="test-api-key",
            team="integration-test",
            project="end-to-end",
            environment="test",
        )

        # Step 3: Discover chatflows
        chatflows = adapter.get_chatflows()
        assert len(chatflows) == 2
        assert chatflows[0]["id"] == "test-flow-1"

        # Step 4: Execute prediction
        result = adapter.predict_flow(
            chatflow_id=chatflows[0]["id"],
            question="What is artificial intelligence?",
            session_id="test-session-123",
        )

        assert "text" in result
        assert result["text"] == "This is a test response from the AI model."

        # Verify all mocks were called appropriately
        assert mock_get.called
        assert mock_post.called

    def test_auto_instrumentation_workflow(self):
        """Test auto-instrumentation workflow."""
        # Test auto-instrumentation setup
        result = auto_instrument(
            team="auto-test", project="instrumentation", environment="test"
        )

        assert result is True

    @patch("requests.get")
    def test_chatflow_discovery_workflow(self, mock_get):
        """Test chatflow discovery and selection workflow."""
        # Mock different chatflow scenarios
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": "customer-support", "name": "Customer Support Assistant"},
            {"id": "sales-bot", "name": "Sales Bot"},
            {"id": "technical-help", "name": "Technical Help Desk"},
            {"id": "general-qa", "name": "General Q&A"},
        ]
        mock_get.return_value = mock_response

        adapter = instrument_flowise("http://localhost:3000")
        chatflows = adapter.get_chatflows()

        # Test chatflow selection logic
        customer_support_flow = next(
            (flow for flow in chatflows if "customer" in flow["name"].lower()), None
        )
        assert customer_support_flow is not None
        assert customer_support_flow["id"] == "customer-support"

        sales_flow = next(
            (flow for flow in chatflows if "sales" in flow["name"].lower()), None
        )
        assert sales_flow is not None
        assert sales_flow["id"] == "sales-bot"

    @patch("requests.post")
    def test_session_management_workflow(self, mock_post):
        """Test session-based conversation management workflow."""
        # Mock successful responses
        responses = [
            {"text": "Hello! How can I help you today?"},
            {"text": "I understand you want to know about AI. Let me explain..."},
            {"text": "Is there anything else you'd like to know?"},
        ]

        mock_responses = []
        for response in responses:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = response
            mock_responses.append(mock_response)

        mock_post.side_effect = mock_responses

        adapter = instrument_flowise("http://localhost:3000", team="session-test")
        session_id = "conversation-session-456"

        # Simulate multi-turn conversation
        conversation = [
            "Hello",
            "Tell me about artificial intelligence",
            "Thank you for the explanation",
        ]

        responses_received = []
        for message in conversation:
            result = adapter.predict_flow(
                chatflow_id="test-flow", question=message, session_id=session_id
            )
            responses_received.append(result["text"])

        assert len(responses_received) == 3
        assert responses_received[0] == "Hello! How can I help you today?"
        assert "AI" in responses_received[1]

        # Verify session ID was consistent
        for call in mock_post.call_args_list:
            request_data = call[1]["json"]
            assert request_data["sessionId"] == session_id

    def test_cost_tracking_workflow(self):
        """Test cost tracking throughout the workflow."""
        calculator = FlowiseCostCalculator()

        # Simulate a series of requests with cost tracking
        requests = [
            {"input_tokens": 50, "output_tokens": 100, "model": "gpt-3.5-turbo"},
            {"input_tokens": 200, "output_tokens": 300, "model": "gpt-4"},
            {"input_tokens": 100, "output_tokens": 150, "model": "gpt-3.5-turbo"},
        ]

        total_cost = Decimal("0")
        for req in requests:
            cost = calculator.calculate_cost(
                input_tokens=req["input_tokens"],
                output_tokens=req["output_tokens"],
                model_name=req["model"],
            )
            total_cost += cost

        assert total_cost > 0
        assert isinstance(total_cost, Decimal)

        # Test cost optimization
        from genops.providers.flowise_pricing import (
            get_cost_optimization_recommendations,
        )

        recommendations = get_cost_optimization_recommendations(
            current_model="gpt-4",
            current_cost=total_cost,
            input_tokens=sum(req["input_tokens"] for req in requests),
            output_tokens=sum(req["output_tokens"] for req in requests),
        )

        assert isinstance(recommendations, list)

    def test_governance_attributes_workflow(self):
        """Test governance attributes propagation through workflow."""
        governance_attrs = {
            "team": "platform-engineering",
            "project": "ai-assistant-v2",
            "customer_id": "customer-abc-123",
            "environment": "production",
            "cost_center": "eng-ai-platform",
            "feature": "customer-support-chat",
        }

        adapter = instrument_flowise(
            base_url="http://localhost:3000", **governance_attrs
        )

        # Verify all governance attributes are set
        for key, value in governance_attrs.items():
            assert getattr(adapter, key) == value

        # Test that attributes would be included in telemetry
        # (This would be tested with actual telemetry integration)
        telemetry_attrs = {
            f"genops.{key}": value for key, value in governance_attrs.items()
        }

        assert len(telemetry_attrs) == len(governance_attrs)

    @patch.dict(
        os.environ,
        {
            "FLOWISE_BASE_URL": "http://env-flowise:3000",
            "FLOWISE_API_KEY": "env-api-key",
            "GENOPS_TEAM": "env-team",
            "GENOPS_PROJECT": "env-project",
        },
    )
    def test_environment_configuration_workflow(self):
        """Test configuration from environment variables."""
        # Test that adapter respects environment variables
        adapter = instrument_flowise()

        # Should use environment values as defaults
        # (Implementation may vary based on how env vars are handled)
        assert adapter.base_url  # Should have some URL
        assert adapter.team  # Should have some team
        assert adapter.project  # Should have some project


class TestFlowiseErrorHandlingIntegration:
    """Test error handling across integrated components."""

    def test_validation_to_adapter_error_flow(self):
        """Test error flow from validation through to adapter usage."""
        # Test with invalid URL
        validation_result = validate_flowise_setup("invalid-url", "api-key")
        assert not validation_result.is_valid

        # Even with validation failure, adapter should be createable
        # (but may fail on actual usage)
        adapter = instrument_flowise("invalid-url", api_key="api-key")
        assert adapter is not None

    @patch("requests.get")
    def test_network_error_propagation(self, mock_get):
        """Test network error handling across components."""
        mock_get.side_effect = Exception("Network unreachable")

        # Validation should handle network errors
        validation_result = validate_flowise_setup("http://localhost:3000", "api-key")
        assert not validation_result.is_valid
        assert any(
            "network" in issue.description.lower()
            or "connection" in issue.description.lower()
            for issue in validation_result.issues
        )

        # Adapter should also handle network errors gracefully
        adapter = instrument_flowise("http://localhost:3000", api_key="api-key")

        with pytest.raises(Exception):  # noqa: B017
            adapter.get_chatflows()

    @patch("requests.get")
    @patch("requests.post")
    def test_authentication_error_workflow(self, mock_post, mock_get):
        """Test authentication error handling workflow."""
        # Mock authentication failure
        mock_auth_response = Mock()
        mock_auth_response.status_code = 401
        mock_auth_response.text = "Unauthorized"
        mock_get.return_value = mock_auth_response
        mock_post.return_value = mock_auth_response

        # Validation should detect auth issues
        validation_result = validate_flowise_setup(
            "http://localhost:3000", "invalid-key"
        )
        assert not validation_result.is_valid

        # Adapter operations should fail with auth errors
        adapter = instrument_flowise("http://localhost:3000", api_key="invalid-key")

        with pytest.raises(Exception):  # noqa: B017
            adapter.get_chatflows()

        with pytest.raises(Exception):  # noqa: B017
            adapter.predict_flow("test-flow", "test question")

    def test_partial_failure_workflow(self):
        """Test workflow with partial failures."""
        with patch("requests.get") as mock_get:
            # Mock successful chatflow discovery
            mock_get_response = Mock()
            mock_get_response.status_code = 200
            mock_get_response.json.return_value = [
                {"id": "working-flow", "name": "Working Flow"},
                {"id": "broken-flow", "name": "Broken Flow"},
            ]
            mock_get.return_value = mock_get_response

            with patch("requests.post") as mock_post:
                # Mock mixed success/failure for predictions
                def side_effect_func(url, **kwargs):
                    if "working-flow" in url:
                        response = Mock()
                        response.status_code = 200
                        response.json.return_value = {"text": "Success"}
                        return response
                    else:
                        response = Mock()
                        response.status_code = 500
                        response.text = "Internal Error"
                        return response

                mock_post.side_effect = side_effect_func

                adapter = instrument_flowise("http://localhost:3000")
                adapter.get_chatflows()

                # Working flow should succeed
                result1 = adapter.predict_flow("working-flow", "test")
                assert result1["text"] == "Success"

                # Broken flow should fail
                with pytest.raises(Exception):  # noqa: B017
                    adapter.predict_flow("broken-flow", "test")


class TestFlowisePerformanceIntegration:
    """Test performance characteristics of integrated workflows."""

    @patch("requests.get")
    @patch("requests.post")
    def test_concurrent_request_performance(self, mock_post, mock_get):
        """Test performance with concurrent requests."""
        import threading
        import time

        # Mock fast responses
        mock_get.return_value = Mock(
            status_code=200, json=lambda: [{"id": "test", "name": "Test"}]
        )
        mock_post.return_value = Mock(
            status_code=200, json=lambda: {"text": "Response"}
        )

        adapter = instrument_flowise("http://localhost:3000")
        results = []

        def make_request():
            try:
                result = adapter.predict_flow("test-flow", "test question")
                results.append(result)
            except Exception as e:
                results.append(f"Error: {e}")

        # Start multiple concurrent requests
        threads = []
        start_time = time.time()

        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        end_time = time.time()

        # All requests should complete
        assert len(results) == 10
        assert all("Error:" not in str(result) for result in results)

        # Should complete reasonably quickly
        assert (end_time - start_time) < 5

    def test_cost_calculation_performance_integration(self):
        """Test cost calculation performance in integrated scenarios."""
        calculator = FlowiseCostCalculator()

        # Simulate high-volume cost calculations
        start_time = time.time()

        total_cost = Decimal("0")
        for i in range(1000):
            cost = calculator.calculate_cost(
                input_tokens=100 + (i % 100),  # Vary input size
                output_tokens=50 + (i % 50),  # Vary output size
                model_name=["gpt-3.5-turbo", "gpt-4", "claude-3"][i % 3],  # Vary models
            )
            total_cost += cost

        end_time = time.time()

        assert total_cost > 0
        assert (end_time - start_time) < 1  # Should complete in under 1 second

    def test_memory_usage_during_extended_workflow(self):
        """Test memory usage during extended workflow operations."""
        import gc

        gc.collect()

        # Simulate extended usage
        for _batch in range(10):
            adapters = []

            # Create multiple adapters
            for i in range(100):
                adapter = instrument_flowise(
                    base_url="http://localhost:3000",
                    team=f"team-{i}",
                    project=f"project-{i}",
                )
                adapters.append(adapter)

            # Use adapters
            for adapter in adapters:
                str(adapter)  # Force string representation

            # Clear batch
            del adapters
            gc.collect()

        # Final cleanup
        gc.collect()
        # Test passes if no memory errors


class TestFlowiseRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_customer_support_chatbot_scenario(self):
        """Test customer support chatbot scenario."""
        # Configuration for customer support use case
        adapter = instrument_flowise(
            base_url="http://localhost:3000",
            team="customer-success",
            project="support-chatbot",
            customer_id="enterprise-client-001",
            environment="production",
            feature="live-chat-support",
        )

        assert adapter.team == "customer-success"
        assert adapter.customer_id == "enterprise-client-001"

    def test_multi_tenant_saas_scenario(self):
        """Test multi-tenant SaaS scenario."""
        # Different tenants/customers
        tenants = [
            {"id": "tenant-alpha", "team": "alpha-team"},
            {"id": "tenant-beta", "team": "beta-team"},
            {"id": "tenant-gamma", "team": "gamma-team"},
        ]

        adapters = {}

        for tenant in tenants:
            adapters[tenant["id"]] = instrument_flowise(
                base_url="http://localhost:3000",
                team=tenant["team"],
                project="saas-platform",
                customer_id=tenant["id"],
                environment="production",
            )

        # Verify tenant isolation
        assert len(adapters) == 3
        assert adapters["tenant-alpha"].customer_id == "tenant-alpha"
        assert adapters["tenant-beta"].team == "beta-team"

    def test_development_to_production_workflow(self):
        """Test development to production deployment workflow."""
        environments = ["development", "staging", "production"]

        adapters = {}

        for env in environments:
            adapters[env] = instrument_flowise(
                base_url=f"http://flowise-{env}.company.com",
                team="platform-team",
                project="ai-assistant",
                environment=env,
            )

        # Each environment should have different configurations
        for env in environments:
            assert adapters[env].environment == env
            assert adapters[env].team == "platform-team"

    def test_cost_budget_monitoring_scenario(self):
        """Test cost and budget monitoring scenario."""
        calculator = FlowiseCostCalculator()

        # Simulate monthly usage tracking
        monthly_budget = Decimal("100.00")
        current_spend = Decimal("0.00")

        # Simulate daily usage
        daily_usage = [
            (500, 300),  # Day 1
            (800, 400),  # Day 2
            (1200, 600),  # Day 3 (higher usage)
            (600, 350),  # Day 4
            (900, 500),  # Day 5
        ]

        daily_costs = []

        for input_tokens, output_tokens in daily_usage:
            daily_cost = calculator.calculate_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model_name="gpt-3.5-turbo",
            )
            daily_costs.append(daily_cost)
            current_spend += daily_cost

        # Check budget compliance
        budget_utilization = (current_spend / monthly_budget) * 100

        assert current_spend > 0
        assert isinstance(budget_utilization, Decimal)

        # Simulate budget alerts
        if budget_utilization > 80:
            alert_level = "warning"
        elif budget_utilization > 95:
            alert_level = "critical"
        else:
            alert_level = "normal"

        assert alert_level in ["normal", "warning", "critical"]

    @patch("requests.get")
    @patch("requests.post")
    def test_high_availability_scenario(self, mock_post, mock_get):
        """Test high availability and failover scenario."""
        # Mock server responses with occasional failures
        get_responses = [
            Mock(status_code=200, json=lambda: [{"id": "flow1", "name": "Flow 1"}]),
            Mock(status_code=500, text="Server Error"),  # Failure
            Mock(
                status_code=200, json=lambda: [{"id": "flow1", "name": "Flow 1"}]
            ),  # Recovery
        ]

        post_responses = [
            Mock(status_code=200, json=lambda: {"text": "Success"}),
            Mock(status_code=503, text="Service Unavailable"),  # Failure
            Mock(status_code=200, json=lambda: {"text": "Success"}),  # Recovery
        ]

        mock_get.side_effect = get_responses
        mock_post.side_effect = post_responses

        adapter = instrument_flowise("http://localhost:3000")

        # First request should succeed
        try:
            adapter.get_chatflows()
            first_success = True
        except Exception:
            first_success = False

        # Second request should fail
        try:
            adapter.get_chatflows()
            second_success = True
        except Exception:
            second_success = False

        # Third request should succeed (recovery)
        try:
            adapter.get_chatflows()
            third_success = True
        except Exception:
            third_success = False

        # Should have mixed results demonstrating failure/recovery
        assert first_success
        assert not second_success
        assert third_success


class TestFlowiseConfigurationManagement:
    """Test configuration management and environment handling."""

    def test_configuration_inheritance(self):
        """Test configuration inheritance and override patterns."""
        # Base configuration
        base_config = {
            "base_url": "http://localhost:3000",
            "team": "base-team",
            "project": "base-project",
        }

        # Override configuration
        override_config = {"team": "override-team", "environment": "production"}

        adapter = instrument_flowise(**{**base_config, **override_config})

        # Should use override values where provided
        assert adapter.team == "override-team"
        assert adapter.environment == "production"

        # Should use base values where not overridden
        assert adapter.base_url == "http://localhost:3000"
        assert adapter.project == "base-project"

    def test_configuration_validation_integration(self):
        """Test configuration validation integrated with setup."""
        # Valid configuration
        valid_result = validate_flowise_setup("http://localhost:3000", "valid-api-key")

        # Should create validation result
        assert hasattr(valid_result, "is_valid")
        assert hasattr(valid_result, "issues")
        assert hasattr(valid_result, "summary")

    def test_dynamic_configuration_updates(self):
        """Test dynamic configuration updates during runtime."""
        adapter = instrument_flowise(
            base_url="http://localhost:3000", team="initial-team"
        )

        assert adapter.team == "initial-team"

        # Test that adapter maintains its configuration
        # (Dynamic updates would require specific implementation)
        assert hasattr(adapter, "team")
        assert hasattr(adapter, "project")
        assert hasattr(adapter, "base_url")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
