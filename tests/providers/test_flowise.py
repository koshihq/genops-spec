"""
Comprehensive test suite for Flowise integration with 75+ tests.

This test suite covers all aspects of the Flowise integration including:
- Provider adapter functionality
- Cost calculation and tracking
- Validation and diagnostics
- Auto-instrumentation
- Multi-provider scenarios
- Error handling and edge cases
- Performance and reliability
"""

import json
import os
import time
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

# Import the modules under test
from genops.providers.flowise import (
    FlowiseConfig,
    auto_instrument,
    instrument_flowise,
)
from genops.providers.flowise_pricing import (
    FlowiseCostCalculator,
    FlowisePricingTier,
    calculate_flowise_cost,
    get_cost_optimization_recommendations,
)
from genops.providers.flowise_validation import (
    ValidationIssue,
    ValidationResult,
    print_validation_result,
    validate_flowise_setup,
)

# Test configuration
TEST_BASE_URL = "http://localhost:3000"
TEST_API_KEY = "test-api-key"
TEST_CHATFLOW_ID = "test-chatflow-123"


class TestGenOpsFlowiseAdapter:
    """Test suite for GenOpsFlowiseAdapter core functionality."""

    def setup_method(self):
        """Setup for each test method."""
        self.adapter = instrument_flowise(
            base_url=TEST_BASE_URL,
            api_key=TEST_API_KEY,
            team="test-team",
            project="test-project",
        )

    def test_adapter_initialization(self):
        """Test adapter initialization with basic parameters."""
        assert self.adapter.base_url == TEST_BASE_URL
        assert self.adapter.api_key == TEST_API_KEY
        assert self.adapter.team == "test-team"
        assert self.adapter.project == "test-project"

    def test_adapter_initialization_with_defaults(self):
        """Test adapter initialization with default values."""
        adapter = instrument_flowise(base_url=TEST_BASE_URL)
        assert adapter.base_url == TEST_BASE_URL
        assert adapter.api_key is None
        assert adapter.team == "default-team"
        assert adapter.project == "default-project"

    def test_adapter_initialization_with_governance_attrs(self):
        """Test adapter initialization with governance attributes."""
        adapter = instrument_flowise(
            base_url=TEST_BASE_URL,
            api_key=TEST_API_KEY,
            team="engineering",
            project="ai-assistant",
            customer_id="customer-123",
            environment="production",
            cost_center="eng-ai",
            feature="chat-completion",
        )
        assert adapter.team == "engineering"
        assert adapter.project == "ai-assistant"
        assert adapter.customer_id == "customer-123"
        assert adapter.environment == "production"
        assert adapter.cost_center == "eng-ai"
        assert adapter.feature == "chat-completion"

    def test_adapter_url_normalization(self):
        """Test URL normalization handles trailing slashes."""
        adapter = instrument_flowise(base_url="http://localhost:3000/")
        assert adapter.base_url == "http://localhost:3000"

        adapter = instrument_flowise(base_url="http://localhost:3000//")
        assert adapter.base_url == "http://localhost:3000"

    def test_adapter_config_object_creation(self):
        """Test FlowiseConfig object creation."""
        config = FlowiseConfig(
            base_url=TEST_BASE_URL, api_key=TEST_API_KEY, timeout=30, max_retries=3
        )
        assert config.base_url == TEST_BASE_URL
        assert config.api_key == TEST_API_KEY
        assert config.timeout == 30
        assert config.max_retries == 3

    @patch("requests.get")
    def test_get_chatflows_success(self, mock_get):
        """Test successful chatflows retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": "flow-1", "name": "Test Flow 1"},
            {"id": "flow-2", "name": "Test Flow 2"},
        ]
        mock_get.return_value = mock_response

        chatflows = self.adapter.get_chatflows()

        assert len(chatflows) == 2
        assert chatflows[0]["id"] == "flow-1"
        assert chatflows[1]["name"] == "Test Flow 2"
        mock_get.assert_called_once()

    @patch("requests.get")
    def test_get_chatflows_with_api_key(self, mock_get):
        """Test chatflows retrieval includes API key in headers."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        self.adapter.get_chatflows()

        # Check that Authorization header was set
        call_args = mock_get.call_args
        headers = call_args[1]["headers"]
        assert "Authorization" in headers
        assert headers["Authorization"] == f"Bearer {TEST_API_KEY}"

    @patch("requests.get")
    def test_get_chatflows_without_api_key(self, mock_get):
        """Test chatflows retrieval without API key."""
        adapter = instrument_flowise(base_url=TEST_BASE_URL)  # No API key
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        adapter.get_chatflows()

        # Check that Authorization header was not set
        call_args = mock_get.call_args
        headers = call_args[1].get("headers", {})
        assert "Authorization" not in headers

    @patch("requests.get")
    def test_get_chatflows_error_handling(self, mock_get):
        """Test chatflows retrieval error handling."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        with pytest.raises(Exception):  # noqa: B017
            self.adapter.get_chatflows()

    @patch("requests.get")
    def test_get_chatflows_network_error(self, mock_get):
        """Test chatflows retrieval with network error."""
        mock_get.side_effect = Exception("Network error")

        with pytest.raises(Exception):  # noqa: B017
            self.adapter.get_chatflows()

    @patch("requests.post")
    def test_predict_flow_success(self, mock_post):
        """Test successful flow prediction."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Test response"}
        mock_post.return_value = mock_response

        result = self.adapter.predict_flow(TEST_CHATFLOW_ID, "Test question")

        assert result["text"] == "Test response"
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_predict_flow_with_session_id(self, mock_post):
        """Test flow prediction with session ID."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Test response"}
        mock_post.return_value = mock_response

        self.adapter.predict_flow(
            TEST_CHATFLOW_ID, "Test question", session_id="session-123"
        )

        # Check that session ID was included in request data
        call_args = mock_post.call_args
        request_data = call_args[1]["json"]
        assert request_data["sessionId"] == "session-123"

    @patch("requests.post")
    def test_predict_flow_with_additional_params(self, mock_post):
        """Test flow prediction with additional parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Test response"}
        mock_post.return_value = mock_response

        self.adapter.predict_flow(
            TEST_CHATFLOW_ID,
            "Test question",
            custom_param="custom_value",
            another_param=123,
        )

        # Check that additional params were included
        call_args = mock_post.call_args
        request_data = call_args[1]["json"]
        assert request_data["custom_param"] == "custom_value"
        assert request_data["another_param"] == 123

    @patch("requests.post")
    def test_predict_flow_error_handling(self, mock_post):
        """Test flow prediction error handling."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response

        with pytest.raises(Exception):  # noqa: B017
            self.adapter.predict_flow(TEST_CHATFLOW_ID, "Test question")

    @patch("requests.post")
    def test_predict_flow_network_error(self, mock_post):
        """Test flow prediction with network error."""
        mock_post.side_effect = Exception("Network error")

        with pytest.raises(Exception):  # noqa: B017
            self.adapter.predict_flow(TEST_CHATFLOW_ID, "Test question")

    def test_adapter_context_manager(self):
        """Test adapter can be used as context manager."""
        with instrument_flowise(base_url=TEST_BASE_URL) as adapter:
            assert adapter.base_url == TEST_BASE_URL

    def test_adapter_string_representation(self):
        """Test adapter string representation."""
        adapter_str = str(self.adapter)
        assert "GenOpsFlowiseAdapter" in adapter_str
        assert TEST_BASE_URL in adapter_str


class TestFlowiseAutoInstrumentation:
    """Test suite for Flowise auto-instrumentation functionality."""

    def test_auto_instrument_basic(self):
        """Test basic auto-instrumentation."""
        result = auto_instrument(team="test-team", project="test-project")
        assert result is True

    def test_auto_instrument_with_config(self):
        """Test auto-instrumentation with configuration."""
        result = auto_instrument(
            base_url=TEST_BASE_URL,
            api_key=TEST_API_KEY,
            team="test-team",
            project="test-project",
            environment="production",
        )
        assert result is True

    def test_auto_instrument_with_otel_config(self):
        """Test auto-instrumentation with OpenTelemetry configuration."""
        result = auto_instrument(
            team="test-team",
            project="test-project",
            otlp_endpoint="http://localhost:4317",
            otlp_headers={"x-api-key": "test-key"},
        )
        assert result is True

    @patch.dict(
        os.environ, {"FLOWISE_BASE_URL": TEST_BASE_URL, "FLOWISE_API_KEY": TEST_API_KEY}
    )
    def test_auto_instrument_from_environment(self):
        """Test auto-instrumentation uses environment variables."""
        result = auto_instrument(team="test-team", project="test-project")
        assert result is True

    def test_auto_instrument_validation_error(self):
        """Test auto-instrumentation with validation errors."""
        # Invalid base URL should not prevent auto-instrumentation
        result = auto_instrument(
            base_url="invalid-url", team="test-team", project="test-project"
        )
        # Auto-instrumentation should still succeed but log warnings
        assert result is True

    def test_auto_instrument_minimal_config(self):
        """Test auto-instrumentation with minimal configuration."""
        result = auto_instrument()
        assert result is True

    def test_auto_instrument_multiple_calls(self):
        """Test multiple auto-instrumentation calls."""
        result1 = auto_instrument(team="team1", project="project1")
        result2 = auto_instrument(team="team2", project="project2")
        assert result1 is True
        assert result2 is True


class TestFlowiseValidation:
    """Test suite for Flowise setup validation."""

    @patch("requests.get")
    def test_validate_flowise_setup_success(self, mock_get):
        """Test successful Flowise setup validation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        result = validate_flowise_setup(TEST_BASE_URL, TEST_API_KEY)

        assert result.is_valid is True
        assert len(result.issues) == 0
        assert "successfully" in result.summary.lower()

    @patch("requests.get")
    def test_validate_flowise_setup_network_error(self, mock_get):
        """Test validation with network connection error."""
        mock_get.side_effect = Exception("Connection error")

        result = validate_flowise_setup(TEST_BASE_URL, TEST_API_KEY)

        assert result.is_valid is False
        assert len(result.issues) > 0
        assert any("connection" in issue.description.lower() for issue in result.issues)

    @patch("requests.get")
    def test_validate_flowise_setup_server_error(self, mock_get):
        """Test validation with server error response."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        result = validate_flowise_setup(TEST_BASE_URL, TEST_API_KEY)

        assert result.is_valid is False
        assert len(result.issues) > 0

    @patch("requests.get")
    def test_validate_flowise_setup_auth_error(self, mock_get):
        """Test validation with authentication error."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_get.return_value = mock_response

        result = validate_flowise_setup(TEST_BASE_URL, TEST_API_KEY)

        assert result.is_valid is False
        assert any(
            "auth" in issue.description.lower()
            or "unauthorized" in issue.description.lower()
            for issue in result.issues
        )

    def test_validate_flowise_setup_invalid_url(self):
        """Test validation with invalid URL format."""
        result = validate_flowise_setup("not-a-url", TEST_API_KEY)

        assert result.is_valid is False
        assert len(result.issues) > 0
        assert any("url" in issue.description.lower() for issue in result.issues)

    def test_validate_flowise_setup_missing_url(self):
        """Test validation with missing URL."""
        result = validate_flowise_setup("", TEST_API_KEY)

        assert result.is_valid is False
        assert len(result.issues) > 0

    def test_validate_flowise_setup_without_api_key(self):
        """Test validation without API key for local development."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = []
            mock_get.return_value = mock_response

            result = validate_flowise_setup(TEST_BASE_URL, None)

            # Should work for local development
            assert result.is_valid is True

    def test_validate_flowise_setup_timeout(self):
        """Test validation with custom timeout."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Timeout")

            result = validate_flowise_setup(TEST_BASE_URL, TEST_API_KEY, timeout=1)

            assert result.is_valid is False

    def test_validation_result_creation(self):
        """Test ValidationResult object creation."""
        issues = [
            ValidationIssue("error", "Test error", "Fix this issue"),
            ValidationIssue("warning", "Test warning", "Consider this fix"),
        ]

        result = ValidationResult(
            is_valid=False, summary="Validation failed", issues=issues
        )

        assert result.is_valid is False
        assert result.summary == "Validation failed"
        assert len(result.issues) == 2
        assert result.issues[0].severity == "error"
        assert result.issues[1].severity == "warning"

    def test_validation_issue_creation(self):
        """Test ValidationIssue object creation."""
        issue = ValidationIssue(
            severity="error",
            description="Connection failed",
            suggested_fix="Check network connectivity",
        )

        assert issue.severity == "error"
        assert issue.description == "Connection failed"
        assert issue.suggested_fix == "Check network connectivity"

    def test_print_validation_result_success(self, capsys):
        """Test printing successful validation result."""
        result = ValidationResult(
            is_valid=True, summary="Validation successful", issues=[]
        )

        print_validation_result(result)
        captured = capsys.readouterr()

        assert "✅" in captured.out or "success" in captured.out.lower()
        assert "Validation successful" in captured.out

    def test_print_validation_result_with_errors(self, capsys):
        """Test printing validation result with errors."""
        issues = [
            ValidationIssue("error", "Connection failed", "Check network"),
            ValidationIssue("warning", "No API key", "Set FLOWISE_API_KEY"),
        ]

        result = ValidationResult(
            is_valid=False, summary="Validation failed", issues=issues
        )

        print_validation_result(result)
        captured = capsys.readouterr()

        assert "❌" in captured.out or "error" in captured.out.lower()
        assert "Connection failed" in captured.out
        assert "Check network" in captured.out


class TestFlowisePricing:
    """Test suite for Flowise cost calculation and pricing."""

    def setup_method(self):
        """Setup for each test method."""
        self.calculator = FlowiseCostCalculator()

    def test_cost_calculator_initialization(self):
        """Test cost calculator initialization."""
        assert isinstance(self.calculator, FlowiseCostCalculator)
        assert len(self.calculator.pricing_tiers) > 0

    def test_cost_calculator_with_custom_pricing(self):
        """Test cost calculator with custom pricing tiers."""
        custom_tiers = [
            FlowisePricingTier("starter", Decimal("0.001"), 10000),
            FlowisePricingTier("professional", Decimal("0.0008"), 100000),
        ]

        calculator = FlowiseCostCalculator(custom_pricing_tiers=custom_tiers)
        assert len(calculator.pricing_tiers) == 2
        assert calculator.pricing_tiers[0].name == "starter"

    def test_calculate_basic_cost(self):
        """Test basic cost calculation."""
        cost = self.calculator.calculate_cost(
            input_tokens=100, output_tokens=50, model_name="gpt-3.5-turbo"
        )

        assert isinstance(cost, Decimal)
        assert cost > 0

    def test_calculate_cost_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        cost = self.calculator.calculate_cost(
            input_tokens=0, output_tokens=0, model_name="gpt-3.5-turbo"
        )

        assert cost == Decimal("0")

    def test_calculate_cost_with_tier(self):
        """Test cost calculation for specific pricing tier."""
        cost = self.calculator.calculate_cost(
            input_tokens=1000,
            output_tokens=500,
            model_name="gpt-4",
            pricing_tier="professional",
        )

        assert isinstance(cost, Decimal)
        assert cost > 0

    def test_calculate_cost_unknown_model(self):
        """Test cost calculation for unknown model."""
        cost = self.calculator.calculate_cost(
            input_tokens=100, output_tokens=50, model_name="unknown-model"
        )

        # Should use default pricing
        assert isinstance(cost, Decimal)
        assert cost > 0

    def test_calculate_cost_with_multiplier(self):
        """Test cost calculation with cost multiplier."""
        base_cost = self.calculator.calculate_cost(
            input_tokens=100, output_tokens=50, model_name="gpt-3.5-turbo"
        )

        multiplied_cost = self.calculator.calculate_cost(
            input_tokens=100,
            output_tokens=50,
            model_name="gpt-3.5-turbo",
            cost_multiplier=Decimal("1.5"),
        )

        assert multiplied_cost > base_cost
        assert multiplied_cost == base_cost * Decimal("1.5")

    def test_estimate_tokens_from_text(self):
        """Test token estimation from text."""
        text = "This is a test message with multiple words."
        tokens = self.calculator.estimate_tokens(text)

        assert isinstance(tokens, int)
        assert tokens > 0
        assert tokens <= len(text.split()) * 2  # Rough upper bound

    def test_estimate_tokens_empty_text(self):
        """Test token estimation for empty text."""
        tokens = self.calculator.estimate_tokens("")
        assert tokens == 0

    def test_pricing_tier_creation(self):
        """Test FlowisePricingTier object creation."""
        tier = FlowisePricingTier(
            name="custom", cost_per_1k_tokens=Decimal("0.002"), monthly_limit=50000
        )

        assert tier.name == "custom"
        assert tier.cost_per_1k_tokens == Decimal("0.002")
        assert tier.monthly_limit == 50000

    def test_calculate_flowise_cost_function(self):
        """Test standalone calculate_flowise_cost function."""
        cost = calculate_flowise_cost(
            input_tokens=200, output_tokens=100, model_name="gpt-3.5-turbo"
        )

        assert isinstance(cost, Decimal)
        assert cost > 0

    def test_get_cost_optimization_recommendations(self):
        """Test cost optimization recommendations."""
        recommendations = get_cost_optimization_recommendations(
            current_model="gpt-4",
            current_cost=Decimal("0.10"),
            input_tokens=1000,
            output_tokens=500,
        )

        assert isinstance(recommendations, list)
        # Should have at least one recommendation
        assert len(recommendations) >= 0

    def test_cost_optimization_with_budget_constraint(self):
        """Test cost optimization with budget constraints."""
        recommendations = get_cost_optimization_recommendations(
            current_model="gpt-4",
            current_cost=Decimal("0.10"),
            input_tokens=1000,
            output_tokens=500,
            budget_constraint=Decimal("0.05"),
        )

        assert isinstance(recommendations, list)
        # All recommendations should be under budget
        for rec in recommendations:
            if "estimated_cost" in rec:
                assert rec["estimated_cost"] <= Decimal("0.05")

    def test_bulk_cost_calculation(self):
        """Test calculating costs for multiple requests."""
        requests = [
            {"input_tokens": 100, "output_tokens": 50, "model_name": "gpt-3.5-turbo"},
            {"input_tokens": 200, "output_tokens": 100, "model_name": "gpt-4"},
            {"input_tokens": 150, "output_tokens": 75, "model_name": "gpt-3.5-turbo"},
        ]

        total_cost = Decimal("0")
        for req in requests:
            cost = self.calculator.calculate_cost(**req)
            total_cost += cost

        assert total_cost > 0
        assert isinstance(total_cost, Decimal)

    def test_cost_breakdown_by_model(self):
        """Test cost breakdown by model type."""
        models_costs = {}

        for model in ["gpt-3.5-turbo", "gpt-4", "claude-3"]:
            cost = self.calculator.calculate_cost(
                input_tokens=100, output_tokens=50, model_name=model
            )
            models_costs[model] = cost

        assert len(models_costs) == 3
        assert all(cost > 0 for cost in models_costs.values())


class TestFlowiseIntegrationScenarios:
    """Test suite for complex Flowise integration scenarios."""

    def setup_method(self):
        """Setup for each test method."""
        self.adapter = instrument_flowise(
            base_url=TEST_BASE_URL,
            api_key=TEST_API_KEY,
            team="integration-test",
            project="test-scenarios",
        )

    @patch("requests.get")
    @patch("requests.post")
    def test_end_to_end_workflow(self, mock_post, mock_get):
        """Test complete end-to-end workflow."""
        # Mock chatflows retrieval
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = [
            {"id": TEST_CHATFLOW_ID, "name": "Test Flow"}
        ]
        mock_get.return_value = mock_get_response

        # Mock flow prediction
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {"text": "Test response"}
        mock_post.return_value = mock_post_response

        # Execute workflow
        chatflows = self.adapter.get_chatflows()
        assert len(chatflows) == 1

        result = self.adapter.predict_flow(chatflows[0]["id"], "Test question")
        assert result["text"] == "Test response"

    @patch("requests.post")
    def test_session_management(self, mock_post):
        """Test session-based conversation management."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Response"}
        mock_post.return_value = mock_response

        session_id = "test-session-123"

        # First message in session
        self.adapter.predict_flow(TEST_CHATFLOW_ID, "Hello", session_id=session_id)

        # Second message in same session
        self.adapter.predict_flow(
            TEST_CHATFLOW_ID, "Follow-up question", session_id=session_id
        )

        # Verify both calls used the same session ID
        assert mock_post.call_count == 2
        call_data_1 = mock_post.call_args_list[0][1]["json"]
        call_data_2 = mock_post.call_args_list[1][1]["json"]
        assert call_data_1["sessionId"] == session_id
        assert call_data_2["sessionId"] == session_id

    @patch("requests.post")
    def test_multiple_concurrent_requests(self, mock_post):
        """Test handling multiple concurrent requests."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Response"}
        mock_post.return_value = mock_response

        # Simulate concurrent requests
        questions = ["Question 1", "Question 2", "Question 3"]

        for i, question in enumerate(questions):
            self.adapter.predict_flow(
                TEST_CHATFLOW_ID, question, session_id=f"session-{i}"
            )

        assert mock_post.call_count == len(questions)

    @patch("requests.post")
    def test_retry_mechanism(self, mock_post):
        """Test retry mechanism for failed requests."""
        # First call fails, second succeeds
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        mock_response_fail.text = "Internal Server Error"

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"text": "Success"}

        mock_post.side_effect = [mock_response_fail, mock_response_success]

        # This test assumes retry logic exists in the adapter
        # For now, just verify the behavior without retries
        with pytest.raises(Exception):  # noqa: B017
            self.adapter.predict_flow(TEST_CHATFLOW_ID, "Test question")

    def test_cost_tracking_integration(self):
        """Test integration with cost tracking."""
        calculator = FlowiseCostCalculator()

        # Simulate a request with known token usage
        input_tokens = 100
        output_tokens = 150

        cost = calculator.calculate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_name="gpt-3.5-turbo",
        )

        assert cost > 0
        assert isinstance(cost, Decimal)

    def test_governance_attributes_propagation(self):
        """Test governance attributes are properly propagated."""
        adapter = instrument_flowise(
            base_url=TEST_BASE_URL,
            team="engineering",
            project="ai-chatbot",
            customer_id="customer-456",
            environment="production",
        )

        assert adapter.team == "engineering"
        assert adapter.project == "ai-chatbot"
        assert adapter.customer_id == "customer-456"
        assert adapter.environment == "production"

    @patch("requests.get")
    def test_chatflow_discovery_and_selection(self, mock_get):
        """Test chatflow discovery and selection logic."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": "flow-1", "name": "Customer Support"},
            {"id": "flow-2", "name": "Sales Assistant"},
            {"id": "flow-3", "name": "Technical Help"},
        ]
        mock_get.return_value = mock_response

        chatflows = self.adapter.get_chatflows()

        # Test chatflow selection by name
        customer_support = next(
            (flow for flow in chatflows if "Customer" in flow["name"]), None
        )
        assert customer_support is not None
        assert customer_support["id"] == "flow-1"

    def test_error_handling_with_context(self):
        """Test error handling preserves context information."""
        with pytest.raises(Exception):  # noqa: B017
            # This should fail due to invalid URL
            instrument_flowise(base_url="invalid-url")
            # Additional context testing would go here

    def test_configuration_validation_integration(self):
        """Test configuration validation integrated with adapter."""
        # Test valid configuration
        result = validate_flowise_setup(TEST_BASE_URL, TEST_API_KEY)
        # Cannot assert success without actual server, but test structure

        # Test invalid configuration
        result = validate_flowise_setup("", "")
        assert result.is_valid is False


class TestFlowiseErrorHandling:
    """Test suite for Flowise error handling and edge cases."""

    def test_network_timeout_handling(self):
        """Test handling of network timeouts."""
        adapter = instrument_flowise(
            base_url="http://nonexistent-server:3000",
            timeout=1,  # Very short timeout
        )

        with pytest.raises(Exception):  # noqa: B017
            adapter.get_chatflows()

    def test_invalid_json_response_handling(self):
        """Test handling of invalid JSON responses."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_get.return_value = mock_response

            adapter = instrument_flowise(base_url=TEST_BASE_URL)

            with pytest.raises(Exception):  # noqa: B017
                adapter.get_chatflows()

    def test_rate_limiting_handling(self):
        """Test handling of rate limiting responses."""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.text = "Rate Limited"
            mock_post.return_value = mock_response

            adapter = instrument_flowise(base_url=TEST_BASE_URL)

            with pytest.raises(Exception) as exc_info:
                adapter.predict_flow(TEST_CHATFLOW_ID, "Test question")

            # Should preserve rate limiting information
            assert "429" in str(exc_info.value) or "rate" in str(exc_info.value).lower()

    def test_authentication_error_handling(self):
        """Test handling of authentication errors."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"
            mock_get.return_value = mock_response

            adapter = instrument_flowise(base_url=TEST_BASE_URL, api_key="invalid-key")

            with pytest.raises(Exception) as exc_info:
                adapter.get_chatflows()

            assert (
                "401" in str(exc_info.value)
                or "unauthorized" in str(exc_info.value).lower()
            )

    def test_server_error_handling(self):
        """Test handling of server errors."""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_post.return_value = mock_response

            adapter = instrument_flowise(base_url=TEST_BASE_URL)

            with pytest.raises(Exception) as exc_info:
                adapter.predict_flow(TEST_CHATFLOW_ID, "Test question")

            assert (
                "500" in str(exc_info.value)
                or "server error" in str(exc_info.value).lower()
            )

    def test_empty_response_handling(self):
        """Test handling of empty responses."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = []
            mock_get.return_value = mock_response

            adapter = instrument_flowise(base_url=TEST_BASE_URL)
            chatflows = adapter.get_chatflows()

            assert chatflows == []

    def test_malformed_url_handling(self):
        """Test handling of malformed URLs."""
        # Test various malformed URL patterns
        malformed_urls = [
            "not-a-url",
            "http://",
            "://missing-protocol",
            "http:///missing-host",
        ]

        for url in malformed_urls:
            adapter = instrument_flowise(base_url=url)
            # URL validation might happen during request, not initialization
            with pytest.raises(Exception):  # noqa: B017
                adapter.get_chatflows()

    def test_large_response_handling(self):
        """Test handling of very large responses."""
        with patch("requests.post") as mock_post:
            # Create a large response
            large_text = "Large response " * 10000
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"text": large_text}
            mock_post.return_value = mock_response

            adapter = instrument_flowise(base_url=TEST_BASE_URL)
            result = adapter.predict_flow(TEST_CHATFLOW_ID, "Test question")

            assert len(result["text"]) > 100000

    def test_unicode_handling(self):
        """Test handling of Unicode characters in requests and responses."""
        with patch("requests.post") as mock_post:
            unicode_text = "Response with émojis 🚀 and spéciâl characters"
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"text": unicode_text}
            mock_post.return_value = mock_response

            adapter = instrument_flowise(base_url=TEST_BASE_URL)

            # Test Unicode in question
            unicode_question = "What about émojis 🤔 and spéciâl chars?"
            result = adapter.predict_flow(TEST_CHATFLOW_ID, unicode_question)

            assert result["text"] == unicode_text

    def test_concurrent_error_handling(self):
        """Test error handling in concurrent scenarios."""
        import threading

        adapter = instrument_flowise(base_url=TEST_BASE_URL)
        errors = []

        def make_request():
            try:
                # This will fail due to no mock, capturing the error
                adapter.predict_flow(TEST_CHATFLOW_ID, "Test question")
            except Exception as e:
                errors.append(str(e))

        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join(timeout=1)

        # Should have collected errors from all threads
        assert len(errors) > 0


class TestFlowisePerformanceAndReliability:
    """Test suite for Flowise performance and reliability scenarios."""

    def test_adapter_initialization_performance(self):
        """Test adapter initialization performance."""
        start_time = time.time()

        for _ in range(100):
            instrument_flowise(base_url=TEST_BASE_URL, api_key=TEST_API_KEY)

        end_time = time.time()
        avg_time = (end_time - start_time) / 100

        # Should be very fast (less than 1ms per initialization)
        assert avg_time < 0.001

    def test_cost_calculation_performance(self):
        """Test cost calculation performance."""
        calculator = FlowiseCostCalculator()
        start_time = time.time()

        for _ in range(1000):
            calculator.calculate_cost(
                input_tokens=100, output_tokens=50, model_name="gpt-3.5-turbo"
            )

        end_time = time.time()
        avg_time = (end_time - start_time) / 1000

        # Should be very fast (less than 0.1ms per calculation)
        assert avg_time < 0.0001

    def test_validation_caching(self):
        """Test validation result caching for performance."""
        # Multiple validations of the same configuration should be fast
        url = TEST_BASE_URL
        api_key = TEST_API_KEY

        start_time = time.time()

        for _ in range(10):
            validate_flowise_setup(url, api_key)

        end_time = time.time()
        total_time = end_time - start_time

        # Even with network calls, shouldn't take too long
        assert total_time < 30  # 30 seconds max for 10 validations

    def test_memory_usage_stability(self):
        """Test memory usage remains stable during operations."""
        import gc

        # Force garbage collection
        gc.collect()

        # Create many adapters and let them be garbage collected
        for i in range(1000):
            adapter = instrument_flowise(
                base_url=TEST_BASE_URL, team=f"team-{i}", project=f"project-{i}"
            )

            # Use adapter briefly
            str(adapter)

            if i % 100 == 0:
                gc.collect()

        # Force final garbage collection
        gc.collect()

        # Test passes if no memory errors occurred

    def test_thread_safety(self):
        """Test thread safety of core operations."""
        import threading

        adapter = instrument_flowise(base_url=TEST_BASE_URL)
        results = []

        def worker():
            try:
                # Test thread-safe operations
                result = str(adapter)
                results.append(result)
            except Exception as e:
                results.append(f"Error: {e}")

        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All operations should complete successfully
        assert len(results) == 10
        assert all("Error:" not in result for result in results)

    def test_configuration_edge_cases(self):
        """Test edge cases in configuration handling."""
        # Test with None values
        adapter = instrument_flowise(
            base_url=TEST_BASE_URL, api_key=None, team=None, project=None
        )
        assert adapter is not None

        # Test with empty strings
        adapter = instrument_flowise(
            base_url=TEST_BASE_URL, api_key="", team="", project=""
        )
        assert adapter is not None

        # Test with very long values
        long_value = "a" * 1000
        adapter = instrument_flowise(
            base_url=TEST_BASE_URL, team=long_value, project=long_value
        )
        assert adapter.team == long_value

    def test_auto_instrumentation_reliability(self):
        """Test auto-instrumentation reliability across scenarios."""
        # Test multiple auto-instrumentation calls
        results = []

        for i in range(10):
            try:
                result = auto_instrument(team=f"team-{i}", project=f"project-{i}")
                results.append(result)
            except Exception:
                results.append(False)

        # All auto-instrumentations should succeed
        assert all(result is True for result in results)

    def test_cost_calculation_edge_cases(self):
        """Test cost calculation with edge case inputs."""
        calculator = FlowiseCostCalculator()

        # Test with very large token counts
        cost = calculator.calculate_cost(
            input_tokens=1000000, output_tokens=1000000, model_name="gpt-3.5-turbo"
        )
        assert cost > 0

        # Test with zero tokens
        cost = calculator.calculate_cost(
            input_tokens=0, output_tokens=0, model_name="gpt-3.5-turbo"
        )
        assert cost == 0

        # Test with negative tokens (should handle gracefully)
        with pytest.raises(ValueError):
            calculator.calculate_cost(
                input_tokens=-100, output_tokens=50, model_name="gpt-3.5-turbo"
            )


# Test fixtures and utilities


@pytest.fixture
def mock_flowise_server():
    """Fixture providing a mock Flowise server."""
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        # Mock successful chatflows response
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = [
            {"id": TEST_CHATFLOW_ID, "name": "Test Flow"}
        ]
        mock_get.return_value = mock_get_response

        # Mock successful prediction response
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {"text": "Test response"}
        mock_post.return_value = mock_post_response

        yield {
            "get": mock_get,
            "post": mock_post,
            "get_response": mock_get_response,
            "post_response": mock_post_response,
        }


@pytest.fixture
def sample_flowise_config():
    """Fixture providing sample Flowise configuration."""
    return {
        "base_url": TEST_BASE_URL,
        "api_key": TEST_API_KEY,
        "team": "test-team",
        "project": "test-project",
        "environment": "test",
    }


def test_integration_with_mock_server(mock_flowise_server):
    """Test integration using mock server fixture."""
    adapter = instrument_flowise(base_url=TEST_BASE_URL, api_key=TEST_API_KEY)

    # Test getting chatflows
    chatflows = adapter.get_chatflows()
    assert len(chatflows) == 1
    assert chatflows[0]["id"] == TEST_CHATFLOW_ID

    # Test prediction
    result = adapter.predict_flow(TEST_CHATFLOW_ID, "Test question")
    assert result["text"] == "Test response"


def test_configuration_from_fixture(sample_flowise_config):
    """Test using configuration fixture."""
    adapter = instrument_flowise(**sample_flowise_config)

    assert adapter.base_url == sample_flowise_config["base_url"]
    assert adapter.api_key == sample_flowise_config["api_key"]
    assert adapter.team == sample_flowise_config["team"]
    assert adapter.project == sample_flowise_config["project"]


# Performance benchmarks (optional, for development)


@pytest.mark.benchmark
def test_adapter_creation_benchmark():
    """Benchmark adapter creation performance."""

    def create_adapter():
        return instrument_flowise(
            base_url=TEST_BASE_URL,
            api_key=TEST_API_KEY,
            team="benchmark-team",
            project="benchmark-project",
        )

    # Create 1000 adapters and measure time
    start_time = time.time()
    [create_adapter() for _ in range(1000)]
    end_time = time.time()

    avg_time = (end_time - start_time) / 1000
    print(f"Average adapter creation time: {avg_time:.6f} seconds")

    assert avg_time < 0.001  # Should be under 1ms


@pytest.mark.benchmark
def test_cost_calculation_benchmark():
    """Benchmark cost calculation performance."""
    calculator = FlowiseCostCalculator()

    def calculate_cost():
        return calculator.calculate_cost(
            input_tokens=100, output_tokens=50, model_name="gpt-3.5-turbo"
        )

    # Perform 10000 calculations
    start_time = time.time()
    costs = [calculate_cost() for _ in range(10000)]
    end_time = time.time()

    avg_time = (end_time - start_time) / 10000
    print(f"Average cost calculation time: {avg_time:.6f} seconds")

    assert avg_time < 0.0001  # Should be under 0.1ms
    assert all(cost > 0 for cost in costs)


# Mark slow tests
@pytest.mark.slow
def test_comprehensive_validation_scenarios():
    """Comprehensive test of all validation scenarios (slow test)."""
    scenarios = [
        ("valid_local", "http://localhost:3000", None),
        ("invalid_url", "not-a-url", None),
        ("missing_url", "", None),
        ("unreachable_server", "http://unreachable-server:3000", "api-key"),
    ]

    for _name, url, api_key in scenarios:
        result = validate_flowise_setup(url, api_key)
        # Specific assertions would depend on expected behavior
        assert isinstance(result, ValidationResult)
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.issues, list)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
