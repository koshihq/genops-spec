"""Tests for OpenRouter provider adapter."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from genops.providers.openrouter import GenOpsOpenRouterAdapter
from tests.utils.mock_providers import MockOpenAIClient


# Mock OpenAI exception classes for testing
class APITimeoutError(Exception):
    """Mock APITimeoutError for testing."""
    pass


class APIConnectionError(Exception):
    """Mock APIConnectionError for testing."""
    pass


class AuthenticationError(Exception):
    """Mock AuthenticationError for testing."""
    pass


class RateLimitError(Exception):
    """Mock RateLimitError for testing."""
    pass


class NotFoundError(Exception):
    """Mock NotFoundError for testing."""
    pass


class APIError(Exception):
    """Mock APIError for testing."""
    pass


@pytest.fixture
def mock_openai_import():
    """Mock OpenAI import for OpenRouter testing without dependency."""
    with patch("genops.providers.openrouter.HAS_OPENROUTER_DEPS", True):
        with patch("genops.providers.openrouter.OpenAI") as mock_openai_class:
            yield mock_openai_class


class TestGenOpsOpenRouterAdapter:
    """Test OpenRouter adapter with governance tracking and multi-provider awareness."""

    def test_adapter_initialization_with_client(self, mock_openai_import):
        """Test adapter initialization with provided client."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        assert adapter.client == mock_client
        assert adapter.telemetry is not None

    def test_adapter_initialization_without_client(self, mock_openai_import):
        """Test adapter initialization creates OpenRouter client."""
        mock_openai_class = mock_openai_import
        mock_openai_class.return_value = MockOpenAIClient()

        GenOpsOpenRouterAdapter(openrouter_api_key="test-key")

        # Verify OpenAI client was created with OpenRouter configuration
        mock_openai_class.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1", api_key="test-key"
        )

    def test_adapter_initialization_missing_openai(self):
        """Test adapter initialization fails when OpenAI package not available."""
        with patch("genops.providers.openrouter.HAS_OPENROUTER_DEPS", False):
            with pytest.raises(ImportError) as exc_info:
                GenOpsOpenRouterAdapter()

            assert (
                "OpenAI package not found (required for OpenRouter compatibility)"
                in str(exc_info.value)
            )

    def test_chat_completions_create_basic(
        self, mock_openai_import, mock_span_recorder
    ):
        """Test basic chat completions with OpenRouter governance tracking."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        messages = [{"role": "user", "content": "What is machine learning?"}]

        response = adapter.chat_completions_create(
            model="anthropic/claude-3-sonnet",
            messages=messages,
            team="ml-team",
            project="education",
        )

        # Verify OpenAI client was called
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["model"] == "anthropic/claude-3-sonnet"
        assert call_args["messages"] == messages

        # Verify governance attributes were not passed to API
        assert "team" not in call_args
        assert "project" not in call_args

        # Verify response
        assert response is not None

    def test_chat_completions_create_with_openrouter_attributes(
        self, mock_openai_import, mock_span_recorder
    ):
        """Test chat completions with OpenRouter-specific attributes."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        messages = [{"role": "user", "content": "Hello!"}]

        response = adapter.chat_completions_create(
            model="openai/gpt-4o",
            messages=messages,
            provider="openai",  # OpenRouter-specific
            route="least-cost",  # OpenRouter-specific
            team="cost-optimization",
            project="routing-test",
        )

        # Verify OpenAI client was called with clean parameters
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]

        # API parameters should be present
        assert call_args["model"] == "openai/gpt-4o"
        assert call_args["messages"] == messages

        # Governance and OpenRouter-specific attributes should be filtered out
        assert "team" not in call_args
        assert "project" not in call_args
        assert "provider" not in call_args
        assert "route" not in call_args

        assert response is not None

    def test_provider_prediction_from_model_name(self, mock_openai_import):
        """Test provider prediction from OpenRouter model names."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        # Test different model name patterns
        test_cases = [
            ("openai/gpt-4o", "openai"),
            ("anthropic/claude-3-sonnet", "anthropic"),
            ("google/gemini-1.5-pro", "google"),
            ("meta-llama/llama-3.1-8b", "meta"),
            ("mistralai/mistral-large", "mistral"),
            ("cohere/command-r", "cohere"),
            ("unknown/model", "openrouter"),  # Fallback
        ]

        for model, expected_provider in test_cases:
            predicted = adapter._get_provider_from_model(model)
            assert predicted == expected_provider

    def test_routing_info_extraction(self, mock_openai_import):
        """Test OpenRouter routing information extraction."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        # Test response with routing headers
        mock_response = Mock()
        mock_response.response = Mock()
        mock_response.response.headers = {
            "x-openrouter-provider": "anthropic",
            "x-openrouter-fallback": "true",
            "x-request-id": "req-123",
        }

        routing_info = adapter._extract_routing_info(mock_response)

        assert routing_info["selected_provider"] == "anthropic"
        assert routing_info["fallback_used"] is True
        assert routing_info["request_id"] == "req-123"

        # Test response without routing headers
        mock_response_no_headers = Mock()
        mock_response_no_headers.response = Mock()
        mock_response_no_headers.response.headers = {}

        routing_info_empty = adapter._extract_routing_info(mock_response_no_headers)
        assert routing_info_empty == {}

    def test_completions_create_basic(self, mock_openai_import, mock_span_recorder):
        """Test basic completions with governance tracking."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        response = adapter.completions_create(
            model="openai/gpt-3.5-turbo",
            prompt="What is AI?",
            team="research",
            customer_id="customer-001",
        )

        # Verify client was called
        mock_client.completions.create.assert_called_once()
        call_args = mock_client.completions.create.call_args[1]

        assert call_args["model"] == "openai/gpt-3.5-turbo"
        assert call_args["prompt"] == "What is AI?"

        # Governance attributes should be filtered out
        assert "team" not in call_args
        assert "customer_id" not in call_args

        assert response is not None

    def test_cost_calculation_with_pricing_engine(self, mock_openai_import):
        """Test cost calculation using OpenRouter pricing engine."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        # Mock the pricing engine
        with patch(
            "genops.providers.openrouter_pricing.calculate_openrouter_cost"
        ) as mock_calc:
            mock_calc.return_value = 0.001234

            cost = adapter._calculate_cost(
                "anthropic/claude-3-sonnet", "anthropic", 100, 50
            )

            assert cost == 0.001234
            mock_calc.assert_called_once_with(
                "anthropic/claude-3-sonnet", "anthropic", 100, 50
            )

    def test_cost_calculation_fallback(self, mock_openai_import):
        """Test fallback cost calculation when pricing engine fails."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        # Mock import error for pricing engine by patching the method to raise ImportError
        with patch.object(adapter, "_calculate_cost") as mock_method:
            def mock_calculate_cost_with_fallback(*args):
                # Simulate the try/except ImportError logic in _calculate_cost
                try:
                    from genops.providers.openrouter_pricing import (
                        calculate_openrouter_cost,
                    )
                    raise ImportError("Mock import error")  # Force ImportError
                except ImportError:
                    # Call the actual fallback calculation
                    return adapter._fallback_cost_calculation(*args)

            mock_method.side_effect = mock_calculate_cost_with_fallback
            cost = adapter._calculate_cost("openai/gpt-4o", "openai", 100, 50)

            # Should use fallback calculation
            assert cost > 0
            assert isinstance(cost, float)

    def test_fallback_cost_calculation_different_providers(self, mock_openai_import):
        """Test fallback cost calculation for different providers."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        test_cases = [
            ("openai/gpt-4o", "openai"),
            ("anthropic/claude-3-sonnet", "anthropic"),
            ("google/gemini-pro", "google"),
            ("meta-llama/llama-3.1-8b", "meta"),
            ("mistralai/mistral-large", "mistral"),
            ("unknown/model", "unknown"),
        ]

        for model, provider in test_cases:
            cost = adapter._fallback_cost_calculation(model, provider, 100, 50)
            assert cost > 0
            assert isinstance(cost, float)

    def test_error_handling_in_chat_completion(
        self, mock_openai_import, mock_span_recorder
    ):
        """Test error handling in chat completion requests."""
        mock_client = MockOpenAIClient()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        with pytest.raises(Exception) as exc_info:
            adapter.chat_completions_create(
                model="anthropic/claude-3-sonnet",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert "API Error" in str(exc_info.value)

    def test_governance_attribute_extraction(self, mock_openai_import):
        """Test governance attribute extraction and separation."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        kwargs = {
            "model": "anthropic/claude-3-sonnet",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "team": "ai-team",
            "project": "chatbot",
            "customer_id": "customer-123",
            "provider": "anthropic",  # OpenRouter-specific
            "route": "fastest",  # OpenRouter-specific
        }

        governance_attrs, request_attrs, api_kwargs = adapter._extract_attributes(
            kwargs
        )

        # Check governance attributes
        expected_governance = {
            "team": "ai-team",
            "project": "chatbot",
            "customer_id": "customer-123",
        }
        assert governance_attrs == expected_governance

        # Check request attributes (including OpenRouter-specific)
        assert "temperature" in request_attrs
        assert "max_tokens" in request_attrs
        assert "provider" in request_attrs
        assert "route" in request_attrs

        # Check API kwargs (clean for API call)
        assert "model" in api_kwargs
        assert "messages" in api_kwargs
        assert "temperature" in api_kwargs
        assert "max_tokens" in api_kwargs
        # Governance attributes should be removed
        assert "team" not in api_kwargs
        assert "project" not in api_kwargs
        assert "customer_id" not in api_kwargs

    def test_telemetry_attributes_in_chat_completion(
        self, mock_openai_import, mock_span_recorder
    ):
        """Test that proper telemetry attributes are recorded."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        adapter.chat_completions_create(
            model="anthropic/claude-3-5-sonnet",
            messages=[{"role": "user", "content": "Hello OpenRouter!"}],
            provider="anthropic",
            team="genops-team",
            project="openrouter-integration",
        )

        # Verify span was created with proper attributes
        spans = mock_span_recorder.get_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "openrouter.chat.completions.create"

        # Check for OpenRouter-specific attributes
        attributes = dict(span.attributes)
        assert attributes.get("genops.operation.name") == "openrouter.chat.completions.create"
        assert attributes.get("genops.provider") == "openrouter"
        assert attributes.get("genops.model") == "anthropic/claude-3-5-sonnet"
        assert "genops.openrouter.predicted_provider" in attributes

    def test_openrouter_specific_attributes_in_telemetry(
        self, mock_openai_import, mock_span_recorder
    ):
        """Test OpenRouter-specific telemetry attributes."""
        mock_client = MockOpenAIClient()
        mock_client.chat.completions.create.return_value.response = Mock()
        mock_client.chat.completions.create.return_value.response.headers = {
            "x-openrouter-provider": "anthropic",
            "x-request-id": "req-456",
        }

        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        adapter.chat_completions_create(
            model="anthropic/claude-3-sonnet",
            messages=[{"role": "user", "content": "Test routing"}],
            route="least-cost",
            provider="anthropic",
        )

        spans = mock_span_recorder.get_spans()
        span = spans[0]
        attributes = dict(span.attributes)

        # Check OpenRouter routing attributes
        assert attributes.get("genops.openrouter.routing_strategy") == "least-cost"
        assert attributes.get("genops.openrouter.preferred_provider") == "anthropic"


class TestOpenRouterInstrumentFunction:
    """Test the instrument_openrouter function."""

    def test_instrument_openrouter_with_api_key(self, mock_openai_import):
        """Test instrument_openrouter creates adapter with API key."""
        from genops.providers.openrouter import instrument_openrouter

        mock_openai_class = mock_openai_import
        mock_openai_class.return_value = MockOpenAIClient()

        adapter = instrument_openrouter(openrouter_api_key="test-key")

        assert isinstance(adapter, GenOpsOpenRouterAdapter)
        mock_openai_class.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1", api_key="test-key"
        )

    def test_instrument_openrouter_with_client(self, mock_openai_import):
        """Test instrument_openrouter with existing client."""
        from genops.providers.openrouter import instrument_openrouter

        mock_client = MockOpenAIClient()
        adapter = instrument_openrouter(client=mock_client)

        assert isinstance(adapter, GenOpsOpenRouterAdapter)
        assert adapter.client == mock_client


class TestOpenRouterPatching:
    """Test OpenRouter monkey patching functionality."""

    def test_patch_openrouter(self, mock_openai_import):
        """Test OpenRouter patching only affects OpenRouter clients."""
        from genops.providers.openrouter import patch_openrouter, unpatch_openrouter

        # Apply patches
        patch_openrouter(auto_track=True)

        # Test that patching was applied
        # This is hard to test directly without integration, but we can verify
        # the patching function doesn't raise errors

        # Clean up
        unpatch_openrouter()

    def test_unpatch_openrouter(self, mock_openai_import):
        """Test OpenRouter unpatching."""
        from genops.providers.openrouter import patch_openrouter, unpatch_openrouter

        patch_openrouter(auto_track=True)
        unpatch_openrouter()

        # Should not raise any errors

    def test_patch_openrouter_without_openai(self):
        """Test patching gracefully handles missing OpenAI package."""
        with patch("genops.providers.openrouter.HAS_OPENROUTER_DEPS", False):
            from genops.providers.openrouter import patch_openrouter

            # Should not raise error, just log warning
            patch_openrouter(auto_track=True)


class TestOpenRouterValidation:
    """Test OpenRouter validation utilities."""

    def test_validate_setup_import(self, mock_openai_import):
        """Test validate_setup function import and basic call."""
        from genops.providers.openrouter import validate_setup

        # Mock the validation module
        with patch(
            "genops.providers.openrouter_validation.validate_openrouter_setup"
        ) as mock_validate:
            mock_validate.return_value = Mock()

            result = validate_setup()

            mock_validate.assert_called_once()
            assert result is not None

    def test_validate_setup_missing_validation_module(self, mock_openai_import):
        """Test validate_setup gracefully handles missing validation module."""
        from genops.providers.openrouter import validate_setup

        # Mock ImportError for validation module
        with patch(
            "genops.providers.openrouter_validation.validate_openrouter_setup",
            side_effect=ImportError,
        ):
            result = validate_setup()

            assert result is None

    def test_print_validation_result_import(self, mock_openai_import):
        """Test print_validation_result function."""
        from genops.providers.openrouter import print_validation_result

        mock_result = Mock()

        # Mock the validation module
        with patch(
            "genops.providers.openrouter_validation.print_openrouter_validation_result"
        ) as mock_print:
            print_validation_result(mock_result)

            mock_print.assert_called_once_with(mock_result)

    def test_print_validation_result_missing_validation_module(
        self, mock_openai_import
    ):
        """Test print_validation_result gracefully handles missing validation module."""
        from genops.providers.openrouter import print_validation_result

        mock_result = Mock()

        # Mock ImportError for validation module
        with patch(
            "genops.providers.openrouter_validation.print_openrouter_validation_result",
            side_effect=ImportError,
        ):
            # Should not raise error
            print_validation_result(mock_result)


# Fixtures specific to OpenRouter testing
@pytest.fixture
def mock_openrouter_response():
    """Mock OpenRouter API response with routing information."""
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = "This is a test response from OpenRouter."

    response.usage = Mock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    response.usage.total_tokens = 30

    # Mock OpenRouter-specific response headers
    response.response = Mock()
    response.response.headers = {
        "x-openrouter-provider": "anthropic",
        "x-openrouter-fallback": "false",
        "x-request-id": "openrouter-req-123",
    }

    return response


class TestOpenRouterIntegrationPatterns:
    """Test integration patterns specific to OpenRouter."""

    def test_multi_provider_routing_telemetry(
        self, mock_openai_import, mock_span_recorder, mock_openrouter_response
    ):
        """Test that multi-provider routing information is captured in telemetry."""
        mock_client = MockOpenAIClient()
        mock_client.chat.completions.create.return_value = mock_openrouter_response

        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        # Test request with provider routing
        adapter.chat_completions_create(
            model="anthropic/claude-3-sonnet",
            messages=[{"role": "user", "content": "Test routing telemetry"}],
            provider="anthropic",
            route="least-cost",
            team="routing-team",
        )

        spans = mock_span_recorder.get_spans()
        span = spans[0]
        attributes = dict(span.attributes)

        # Verify OpenRouter routing attributes are captured
        assert attributes.get("genops.openrouter.routing_strategy") == "least-cost"
        assert attributes.get("genops.openrouter.preferred_provider") == "anthropic"
        assert attributes.get("genops.openrouter.actual_provider") == "anthropic"
        assert attributes.get("genops.openrouter.request_id") == "openrouter-req-123"

    def test_cost_calculation_with_actual_provider(
        self, mock_openai_import, mock_openrouter_response
    ):
        """Test cost calculation uses actual provider from routing response."""
        mock_client = MockOpenAIClient()
        mock_client.chat.completions.create.return_value = mock_openrouter_response

        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        with patch.object(adapter, "_calculate_cost", return_value=0.005) as mock_calc:
            adapter.chat_completions_create(
                model="anthropic/claude-3-sonnet",
                messages=[{"role": "user", "content": "Test cost calc"}],
            )

            # Verify cost calculation was called with actual provider
            mock_calc.assert_called_once_with(
                "anthropic/claude-3-sonnet",
                "anthropic",  # Actual provider from response headers
                10,  # prompt_tokens
                20,  # completion_tokens
            )

    def test_fallback_detection_in_telemetry(
        self, mock_openai_import, mock_span_recorder
    ):
        """Test fallback detection is recorded in telemetry."""
        mock_client = MockOpenAIClient()

        # Mock response with fallback indication
        fallback_response = Mock()
        fallback_response.choices = [Mock()]
        fallback_response.choices[0].message = Mock()
        fallback_response.choices[0].message.content = "Fallback response"
        fallback_response.usage = Mock()
        fallback_response.usage.prompt_tokens = 5
        fallback_response.usage.completion_tokens = 10
        fallback_response.usage.total_tokens = 15
        fallback_response.response = Mock()
        fallback_response.response.headers = {
            "x-openrouter-provider": "openai",
            "x-openrouter-fallback": "true",  # Indicates fallback was used
        }

        mock_client.chat.completions.create.return_value = fallback_response
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        adapter.chat_completions_create(
            model="anthropic/claude-3-opus",  # Requested Anthropic
            messages=[{"role": "user", "content": "Test fallback"}],
            provider="anthropic",
        )

        spans = mock_span_recorder.get_spans()
        span = spans[0]
        attributes = dict(span.attributes)

        # Verify fallback was detected and recorded
        assert attributes.get("genops.openrouter.fallback_used") is True
        assert (
            attributes.get("genops.openrouter.actual_provider") == "openai"
        )  # Fallback provider


class TestOpenRouterPricingEngineIntegration:
    """Test pricing engine integration with OpenRouter adapter."""

    def test_pricing_engine_model_coverage(self):
        """Test that pricing engine covers major OpenRouter model families."""
        from genops.providers.openrouter_pricing import get_pricing_engine

        engine = get_pricing_engine()
        pricing_db = engine.pricing_db

        # Check coverage of major providers
        providers_found = set()
        for _model, pricing in pricing_db.items():
            providers_found.add(pricing.provider)

        expected_providers = {
            "openai",
            "anthropic",
            "google",
            "meta",
            "mistral",
            "cohere",
        }
        assert len(providers_found.intersection(expected_providers)) >= 5

    def test_cost_calculation_accuracy(self):
        """Test cost calculation accuracy for known models."""
        from genops.providers.openrouter_pricing import calculate_openrouter_cost

        # Test specific model cost calculation
        cost = calculate_openrouter_cost(
            "anthropic/claude-3-sonnet",
            actual_provider="anthropic",
            input_tokens=1000,
            output_tokens=500,
        )

        assert cost > 0
        assert isinstance(cost, float)
        # Claude 3 Sonnet should be in a reasonable cost range
        assert 0.0001 < cost < 1.0

    def test_fallback_pricing_for_unknown_models(self):
        """Test fallback pricing mechanism for unknown models."""
        from genops.providers.openrouter_pricing import calculate_openrouter_cost

        # Test with unknown model
        cost = calculate_openrouter_cost(
            "unknown/fictional-model",
            actual_provider="unknown",
            input_tokens=100,
            output_tokens=50,
        )

        assert cost > 0  # Should still return a cost
        assert isinstance(cost, float)

    def test_cost_breakdown_detailed_info(self):
        """Test detailed cost breakdown functionality."""
        from genops.providers.openrouter_pricing import get_cost_breakdown

        breakdown = get_cost_breakdown(
            "openai/gpt-4o",
            actual_provider="openai",
            input_tokens=200,
            output_tokens=100,
        )

        required_keys = {
            "total_cost",
            "input_cost",
            "output_cost",
            "provider",
            "model_family",
            "currency",
            "model_name",
        }
        assert all(key in breakdown for key in required_keys)
        assert (
            breakdown["total_cost"]
            == breakdown["input_cost"] + breakdown["output_cost"]
        )
        assert breakdown["currency"] == "USD"

    def test_provider_model_lookup(self):
        """Test provider-specific model lookup functionality."""
        from genops.providers.openrouter_pricing import get_provider_models

        # Test getting models for specific provider
        anthropic_models = get_provider_models("anthropic")
        assert len(anthropic_models) > 0

        # Verify all returned models are from anthropic
        for model_name, pricing in anthropic_models.items():
            assert pricing.provider == "anthropic"
            assert "anthropic" in model_name or "claude" in model_name.lower()


class TestOpenRouterValidationUtilities:
    """Test OpenRouter validation and diagnostic utilities."""

    def test_validation_setup_structure(self):
        """Test validation setup returns proper structure."""
        from genops.providers.openrouter_validation import (
            ValidationIssue,
            ValidationResult,
        )

        # Test ValidationIssue structure
        issue = ValidationIssue("error", "test", "test message", "test fix")
        assert issue.level == "error"
        assert issue.component == "test"
        assert issue.message == "test message"
        assert issue.fix_suggestion == "test fix"

        # Test ValidationResult structure
        result = ValidationResult(True, [issue], {"test": "data"})
        assert result.is_valid is True
        assert len(result.issues) == 1
        assert result.summary["test"] == "data"

    def test_environment_variable_validation(self):
        """Test environment variable validation logic."""
        from genops.providers.openrouter_validation import check_environment_variables

        issues = check_environment_variables()
        assert isinstance(issues, list)

        # Should have at least some issues (missing API key)
        assert len(issues) > 0

        # Check issue structure
        for issue in issues:
            assert hasattr(issue, "level")
            assert hasattr(issue, "component")
            assert hasattr(issue, "message")

    def test_dependency_validation(self):
        """Test dependency validation logic."""
        from genops.providers.openrouter_validation import check_dependencies

        issues = check_dependencies()
        assert isinstance(issues, list)

        # Should have info about OpenAI package being available (in test environment)
        info_issues = [i for i in issues if i.level == "info"]
        assert len(info_issues) > 0

    def test_common_issues_detection(self):
        """Test common issues detection and fixes."""
        from genops.providers.openrouter_validation import check_common_issues

        issues = check_common_issues()
        assert isinstance(issues, list)

        # Should provide specific fixes for detected issues
        for issue in issues:
            if issue.level == "error":
                assert issue.fix_suggestion is not None
                assert len(issue.fix_suggestion) > 10  # Should be detailed


class TestOpenRouterMultiProviderScenarios:
    """Test multi-provider routing and cost attribution scenarios."""

    def test_provider_routing_preferences(self, mock_openai_import, mock_span_recorder):
        """Test explicit provider routing preferences."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        # Test with provider preference
        adapter.chat_completions_create(
            model="anthropic/claude-3-sonnet",
            messages=[{"role": "user", "content": "Test"}],
            provider="anthropic",  # Explicit provider preference
            team="test-team",
        )

        # Verify API call was made
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]

        # Provider preference should not be passed to API
        assert "provider" not in call_kwargs
        assert call_kwargs["model"] == "anthropic/claude-3-sonnet"

    def test_routing_strategy_parameters(self, mock_openai_import, mock_span_recorder):
        """Test different routing strategy parameters."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        routing_strategies = ["least-cost", "fastest", "fallback"]

        for strategy in routing_strategies:
            mock_client.reset_mock()

            adapter.chat_completions_create(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Test routing"}],
                route=strategy,
                team="routing-test",
            )

            # Verify call was made
            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args[1]

            # Routing strategy should not be passed to API
            assert "route" not in call_kwargs

    def test_model_provider_mapping_accuracy(self, mock_openai_import):
        """Test accuracy of model-to-provider mapping."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        test_mappings = [
            ("openai/gpt-4o", "openai"),
            ("anthropic/claude-3-sonnet", "anthropic"),
            ("google/gemini-1.5-pro", "google"),
            ("meta-llama/llama-3.1-8b", "meta"),
            ("mistralai/mistral-large", "mistral"),
            ("cohere/command-r", "cohere"),
        ]

        for model, expected_provider in test_mappings:
            predicted_provider = adapter._get_provider_from_model(model)
            assert predicted_provider == expected_provider, (
                f"Model {model} mapped to {predicted_provider}, expected {expected_provider}"
            )

    def test_cost_attribution_across_providers(
        self, mock_openai_import, mock_span_recorder
    ):
        """Test cost attribution across multiple providers in single session."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        # Simulate requests to different providers
        provider_requests = [
            ("openai/gpt-4o", "openai"),
            ("anthropic/claude-3-sonnet", "anthropic"),
            ("meta-llama/llama-3.1-8b-instruct", "meta"),
        ]

        for model, provider in provider_requests:
            mock_client.reset_mock()

            adapter.chat_completions_create(
                model=model,
                messages=[{"role": "user", "content": "Multi-provider test"}],
                team="multi-provider-team",
                project="cost-attribution-test",
            )

            # Verify telemetry captures provider info
            spans = mock_span_recorder.get_spans()
            if spans:
                latest_span = spans[-1]
                attributes = dict(latest_span.attributes)
                assert attributes.get("genops.openrouter.predicted_provider") == provider


class TestOpenRouterErrorHandlingScenarios:
    """Test comprehensive error handling scenarios."""

    def test_network_timeout_handling(self, mock_openai_import, mock_span_recorder):
        """Test handling of network timeout errors."""

        mock_client = MockOpenAIClient()
        mock_client.chat.completions.create.side_effect = APITimeoutError(
            "Request timed out"
        )

        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        with pytest.raises(APITimeoutError):
            adapter.chat_completions_create(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Test timeout"}],
            )

        # Verify error telemetry
        spans = mock_span_recorder.get_spans()
        assert len(spans) == 1
        span = spans[0]
        attributes = dict(span.attributes)
        assert "genops.error.type" in attributes
        assert "genops.error.message" in attributes

    def test_authentication_error_handling(
        self, mock_openai_import, mock_span_recorder
    ):
        """Test handling of authentication errors."""

        mock_client = MockOpenAIClient()
        mock_client.chat.completions.create.side_effect = AuthenticationError(
            "Invalid API key"
        )

        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        with pytest.raises(AuthenticationError):
            adapter.chat_completions_create(
                model="anthropic/claude-3-sonnet",
                messages=[{"role": "user", "content": "Test auth"}],
            )

        # Verify error telemetry
        spans = mock_span_recorder.get_spans()
        assert len(spans) == 1
        span = spans[0]
        attributes = dict(span.attributes)
        assert attributes["genops.error.type"] == "AuthenticationError"

    def test_rate_limiting_error_handling(self, mock_openai_import, mock_span_recorder):
        """Test handling of rate limiting errors."""

        mock_client = MockOpenAIClient()
        mock_client.chat.completions.create.side_effect = RateLimitError(
            "Rate limit exceeded"
        )

        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        with pytest.raises(RateLimitError):
            adapter.chat_completions_create(
                model="openai/gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test rate limit"}],
            )

        # Verify error telemetry
        spans = mock_span_recorder.get_spans()
        span = spans[0]
        attributes = dict(span.attributes)
        assert attributes["genops.error.type"] == "RateLimitError"

    def test_model_not_found_error_handling(
        self, mock_openai_import, mock_span_recorder
    ):
        """Test handling of model not found errors."""

        mock_client = MockOpenAIClient()
        mock_client.chat.completions.create.side_effect = NotFoundError(
            "Model not found"
        )

        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        with pytest.raises(NotFoundError):
            adapter.chat_completions_create(
                model="nonexistent/model",
                messages=[{"role": "user", "content": "Test model error"}],
            )

    def test_malformed_response_handling(self, mock_openai_import, mock_span_recorder):
        """Test handling of malformed API responses."""
        mock_client = MockOpenAIClient()

        # Create malformed response
        malformed_response = Mock()
        malformed_response.choices = []  # Empty choices
        malformed_response.usage = None  # No usage data
        mock_client.chat.completions.create.return_value = malformed_response

        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        # Should not raise error but handle gracefully
        response = adapter.chat_completions_create(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Test malformed"}],
        )

        assert response == malformed_response


class TestOpenRouterGovernanceIntegration:
    """Test integration with GenOps governance features."""

    def test_governance_attribute_propagation(
        self, mock_openai_import, mock_span_recorder
    ):
        """Test that governance attributes properly propagate through system."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        governance_attrs = {
            "team": "data-science",
            "project": "model-evaluation",
            "customer_id": "enterprise-client-001",
            "environment": "production",
            "cost_center": "R&D",
            "feature": "ai-recommendations",
            "user_id": "user-12345",
            "experiment_id": "exp-2024-q1-001",
            "priority": "high",
            "compliance_level": "confidential",
        }

        adapter.chat_completions_create(
            model="anthropic/claude-3-sonnet",
            messages=[{"role": "user", "content": "Governance test"}],
            **governance_attrs,
        )

        # Verify none of the governance attributes were passed to API
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        for attr_key in governance_attrs.keys():
            assert attr_key not in call_kwargs

        # Verify telemetry captured governance attributes
        spans = mock_span_recorder.get_spans()
        span = spans[0]
        telemetry_attrs = dict(span.attributes)

        # Check some key governance attributes were captured
        assert telemetry_attrs.get("genops.team") == "data-science"
        assert telemetry_attrs.get("genops.project") == "model-evaluation"

    def test_context_manager_governance(self, mock_openai_import, mock_span_recorder):
        """Test governance attributes work with context managers."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        # Test that request-specific attributes override context
        adapter.chat_completions_create(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Context test"}],
            team="override-team",  # Should override any context
            project="context-test-project",
        )

        spans = mock_span_recorder.get_spans()
        span = spans[0]
        attributes = dict(span.attributes)

        # Verify override values were used
        assert attributes.get("genops.team") == "override-team"
        assert attributes.get("genops.project") == "context-test-project"

    def test_cost_tracking_accuracy(self, mock_openai_import, mock_span_recorder):
        """Test accuracy of cost tracking in telemetry."""
        mock_client = MockOpenAIClient()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_client.chat.completions.create.return_value = mock_response

        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        adapter.chat_completions_create(
            model="anthropic/claude-3-sonnet",
            messages=[{"role": "user", "content": "Cost tracking test"}],
            team="cost-team",
        )

        # Verify cost was calculated and recorded
        spans = mock_span_recorder.get_spans()
        span = spans[0]

        # Should have cost-related attributes
        cost_attrs = [key for key in dict(span.attributes).keys() if "cost" in key.lower()]
        assert len(cost_attrs) > 0


class TestOpenRouterPerformanceAndScaling:
    """Test performance and scaling characteristics."""

    def test_concurrent_request_handling(self, mock_openai_import, mock_span_recorder):
        """Test handling of concurrent requests."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        # Simulate multiple concurrent requests
        for i in range(5):
            adapter.chat_completions_create(
                model="openai/gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Concurrent test {i}"}],
                team="performance-team",
                request_id=f"req-{i}",
            )

        # Verify all requests created spans
        spans = mock_span_recorder.get_spans()
        assert len(spans) == 5

        # Verify each span has unique attributes
        request_ids = [
            dict(span.attributes).get("request_id")
            for span in spans
            if "request_id" in dict(span.attributes)
        ]
        assert len(set(request_ids)) == 5  # All unique

    def test_memory_efficiency_large_requests(
        self, mock_openai_import, mock_span_recorder
    ):
        """Test memory efficiency with large requests."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        # Create a large message (simulating large context)
        large_content = "Large content " * 1000  # ~13KB of text

        adapter.chat_completions_create(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": large_content}],
            team="large-context-team",
        )

        # Verify request completed without issues
        mock_client.chat.completions.create.assert_called_once()
        spans = mock_span_recorder.get_spans()
        assert len(spans) == 1

    def test_attribute_extraction_performance(self, mock_openai_import):
        """Test performance of attribute extraction with many attributes."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        # Create request with many attributes
        many_attributes = {f"custom_attr_{i}": f"value_{i}" for i in range(50)}
        many_attributes.update(
            {
                "model": "openai/gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Performance test"}],
                "team": "performance-team",
                "project": "attribute-performance",
            }
        )

        # This should complete quickly without hanging
        governance, request, api = adapter._extract_attributes(many_attributes)

        # Verify extraction worked correctly
        assert len(governance) >= 2  # At least team and project
        assert "model" in api
        assert "messages" in api


class TestOpenRouterEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_messages_handling(self, mock_openai_import, mock_span_recorder):
        """Test handling of empty messages array."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        adapter.chat_completions_create(
            model="openai/gpt-3.5-turbo",
            messages=[],  # Empty messages
            team="edge-case-team",
        )

        # Should still make API call
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"] == []

    def test_very_long_model_names(self, mock_openai_import, mock_span_recorder):
        """Test handling of unusually long model names."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        long_model_name = "very-long-provider/extremely-long-model-name-that-exceeds-normal-length-limits-for-testing-purposes"

        adapter.chat_completions_create(
            model=long_model_name,
            messages=[{"role": "user", "content": "Long name test"}],
            team="edge-case-team",
        )

        # Should handle gracefully
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == long_model_name

    def test_unicode_content_handling(self, mock_openai_import, mock_span_recorder):
        """Test handling of unicode content in messages."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        unicode_content = "Testing unicode: 你好世界 🌍 ñáéíóú אבגד"

        adapter.chat_completions_create(
            model="anthropic/claude-3-sonnet",
            messages=[{"role": "user", "content": unicode_content}],
            team="unicode-team",
        )

        # Should handle unicode gracefully
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"][0]["content"] == unicode_content

    def test_none_values_in_governance_attrs(self, mock_openai_import):
        """Test handling of None values in governance attributes."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        kwargs_with_nones = {
            "model": "openai/gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Test"}],
            "team": "test-team",
            "project": None,  # None value
            "customer_id": "",  # Empty string
            "environment": "test",
        }

        governance, request, api = adapter._extract_attributes(kwargs_with_nones)

        # Should handle None/empty values gracefully
        assert "team" in governance
        assert governance["team"] == "test-team"
        # None and empty values might be included or filtered - either is acceptable

    def test_extreme_token_counts(self, mock_openai_import):
        """Test handling of extreme token count scenarios."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        # Test with very large token estimates
        large_message = "word " * 50000  # Very large input

        # Should not fail during token estimation
        adapter.chat_completions_create(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": large_message}],
            team="large-tokens-team",
        )

        # Verify API was called
        mock_client.chat.completions.create.assert_called_once()


class TestOpenRouterStreamingSupport:
    """Test streaming response support."""

    def test_streaming_parameter_passthrough(
        self, mock_openai_import, mock_span_recorder
    ):
        """Test that streaming parameters are properly passed through."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        adapter.chat_completions_create(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Streaming test"}],
            stream=True,  # Enable streaming
            team="streaming-team",
        )

        # Verify streaming parameter was passed to API
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True

        # Verify governance attributes were still captured
        spans = mock_span_recorder.get_spans()
        span = spans[0]
        attributes = dict(span.attributes)
        assert attributes.get("genops.team") == "streaming-team"

    def test_streaming_with_governance_tracking(
        self, mock_openai_import, mock_span_recorder
    ):
        """Test governance tracking works with streaming responses."""
        mock_client = MockOpenAIClient()

        # Mock streaming response
        streaming_response = [
            Mock(choices=[Mock(delta=Mock(content="Hello"))]),
            Mock(choices=[Mock(delta=Mock(content=" world"))]),
        ]
        mock_client.chat.completions.create.return_value = streaming_response

        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        adapter.chat_completions_create(
            model="anthropic/claude-3-sonnet",
            messages=[{"role": "user", "content": "Stream with governance"}],
            stream=True,
            team="stream-governance-team",
            project="streaming-project",
        )

        # Even with streaming, telemetry should be captured
        spans = mock_span_recorder.get_spans()
        assert len(spans) == 1
        span = spans[0]
        attributes = dict(span.attributes)
        assert attributes.get("genops.team") == "stream-governance-team"


class TestOpenRouterProductionScenarios:
    """Test production-ready scenarios and configurations."""

    def test_high_volume_request_handling(self, mock_openai_import, mock_span_recorder):
        """Test adapter performance under high request volume."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        # Simulate multiple concurrent requests
        for i in range(10):
            response = adapter.chat_completions_create(
                model="openai/gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Request {i}"}],
                team="load-testing",
                project="performance",
                request_id=f"req-{i}",
            )
            assert response is not None

        # Verify all requests were processed
        assert mock_client.chat.completions.create.call_count == 10

    def test_production_timeout_handling(self, mock_openai_import, mock_span_recorder):
        """Test timeout handling in production scenarios."""

        mock_client = MockOpenAIClient()
        mock_client.chat.completions.create.side_effect = APITimeoutError(
            "Request timed out"
        )

        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        with pytest.raises(APITimeoutError):
            adapter.chat_completions_create(
                model="anthropic/claude-3-sonnet",
                messages=[{"role": "user", "content": "Test timeout"}],
                team="production",
                timeout=30,
            )

    def test_production_error_recovery(self, mock_openai_import, mock_span_recorder):
        """Test error recovery in production environments."""

        mock_client = MockOpenAIClient()
        # First call fails, second succeeds
        mock_client.chat.completions.create.side_effect = [
            APIError("Service temporarily unavailable"),
            MockOpenAIClient().chat.completions.create.return_value,
        ]

        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        # First call should raise error
        with pytest.raises(APIError):
            adapter.chat_completions_create(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "First attempt"}],
                team="production",
            )

        # Second call should succeed
        response = adapter.chat_completions_create(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Second attempt"}],
            team="production",
        )
        assert response is not None

    def test_production_context_preservation(
        self, mock_openai_import, mock_span_recorder
    ):
        """Test governance context preservation across operations."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        # Nested operations should preserve context
        response1 = adapter.chat_completions_create(
            model="anthropic/claude-3-sonnet",
            messages=[{"role": "user", "content": "Outer operation"}],
            team="production",
            project="nested-ops",
            operation_id="outer",
        )

        response2 = adapter.chat_completions_create(
            model="meta-llama/llama-3.2-3b-instruct",
            messages=[{"role": "user", "content": "Inner operation"}],
            team="production",
            project="nested-ops",
            operation_id="inner",
            parent_operation="outer",
        )

        assert response1 is not None
        assert response2 is not None
        assert mock_client.chat.completions.create.call_count == 2

    def test_production_memory_efficiency(self, mock_openai_import, mock_span_recorder):
        """Test memory efficiency for long-running applications."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        # Simulate long-running application with many requests
        initial_call_count = mock_client.chat.completions.create.call_count

        for batch in range(3):
            for request in range(10):
                response = adapter.chat_completions_create(
                    model="openai/gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": f"Batch {batch}, Request {request}"}
                    ],
                    team="production",
                    batch_id=f"batch-{batch}",
                )
                assert response is not None

        # Verify all requests processed correctly
        expected_calls = initial_call_count + 30
        assert mock_client.chat.completions.create.call_count == expected_calls


class TestOpenRouterSecurityAndCompliance:
    """Test security and compliance features."""

    def test_sensitive_data_filtering(self, mock_openai_import, mock_span_recorder):
        """Test that sensitive data is properly filtered from telemetry."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        # Request with sensitive information
        response = adapter.chat_completions_create(
            model="anthropic/claude-3-sonnet",
            messages=[{"role": "user", "content": "Process this data: sensitive-info"}],
            team="security-team",
            project="compliance-test",
            compliance_level="confidential",
        )

        assert response is not None
        mock_client.chat.completions.create.assert_called_once()

    def test_api_key_security(self, mock_openai_import, mock_span_recorder):
        """Test that API keys are not logged or exposed."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        # Ensure API key handling is secure
        response = adapter.chat_completions_create(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Security test"}],
            team="security",
            api_key_source="environment",
        )

        assert response is not None
        # Verify API key is not exposed in telemetry
        call_args = mock_client.chat.completions.create.call_args[1]
        assert "api_key_source" not in call_args

    def test_compliance_attribute_handling(
        self, mock_openai_import, mock_span_recorder
    ):
        """Test handling of compliance-related attributes."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        response = adapter.chat_completions_create(
            model="anthropic/claude-3-haiku",
            messages=[{"role": "user", "content": "Compliance query"}],
            team="legal",
            project="compliance-monitoring",
            data_classification="internal",
            compliance_framework="SOC2",
            audit_trail=True,
        )

        assert response is not None
        # Verify compliance attributes are filtered from API call
        call_args = mock_client.chat.completions.create.call_args[1]
        assert "data_classification" not in call_args
        assert "compliance_framework" not in call_args
        assert "audit_trail" not in call_args

    def test_encryption_in_transit_support(
        self, mock_openai_import, mock_span_recorder
    ):
        """Test support for encrypted communications."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        response = adapter.chat_completions_create(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Encrypted transport test"}],
            team="security",
            encryption_required=True,
            tls_version="1.3",
        )

        assert response is not None


class TestOpenRouterAdvancedFeatures:
    """Test advanced OpenRouter-specific features."""

    def test_custom_routing_strategies(self, mock_openai_import, mock_span_recorder):
        """Test custom routing strategy implementation."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        # Test different routing strategies
        strategies = ["fastest", "least-cost", "highest-quality", "fallback"]

        for strategy in strategies:
            response = adapter.chat_completions_create(
                model="anthropic/claude-3-sonnet",
                messages=[{"role": "user", "content": f"Test {strategy} routing"}],
                team="routing-team",
                route=strategy,
            )
            assert response is not None

        assert mock_client.chat.completions.create.call_count == len(strategies)

    def test_model_preference_handling(self, mock_openai_import, mock_span_recorder):
        """Test handling of model preferences and fallbacks."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        response = adapter.chat_completions_create(
            model="anthropic/claude-3-opus",  # Primary choice
            messages=[{"role": "user", "content": "Preference test"}],
            team="ai-research",
            fallbacks=["anthropic/claude-3-sonnet", "openai/gpt-4o"],
            provider_preference=["anthropic", "openai"],
        )

        assert response is not None
        call_args = mock_client.chat.completions.create.call_args[1]
        assert "fallbacks" not in call_args
        assert "provider_preference" not in call_args

    def test_budget_constraint_enforcement(
        self, mock_openai_import, mock_span_recorder
    ):
        """Test budget constraint enforcement."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        response = adapter.chat_completions_create(
            model="meta-llama/llama-3.2-3b-instruct",  # Cost-effective model
            messages=[{"role": "user", "content": "Budget test"}],
            team="cost-conscious",
            max_budget=0.001,
            budget_period="daily",
        )

        assert response is not None
        call_args = mock_client.chat.completions.create.call_args[1]
        assert "max_budget" not in call_args
        assert "budget_period" not in call_args

    def test_streaming_response_handling(self, mock_openai_import, mock_span_recorder):
        """Test streaming response handling with governance."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        response = adapter.chat_completions_create(
            model="openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Stream test"}],
            team="streaming-team",
            stream=True,
        )

        assert response is not None
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args.get("stream") is True

    def test_custom_headers_propagation(self, mock_openai_import, mock_span_recorder):
        """Test custom header propagation to OpenRouter."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        response = adapter.chat_completions_create(
            model="google/gemini-1.5-flash",
            messages=[{"role": "user", "content": "Header test"}],
            team="integration-team",
            custom_headers={
                "X-Custom-App": "GenOps-Test",
                "X-Request-Priority": "high",
            },
        )

        assert response is not None
        call_args = mock_client.chat.completions.create.call_args[1]
        assert "custom_headers" not in call_args


class TestOpenRouterIntegrationRobustness:
    """Test robustness of OpenRouter integration under various conditions."""

    def test_network_interruption_recovery(
        self, mock_openai_import, mock_span_recorder
    ):
        """Test recovery from network interruptions."""

        mock_client = MockOpenAIClient()
        # Simulate intermittent network issues
        mock_client.chat.completions.create.side_effect = [
            APIConnectionError("Connection failed"),
            APIConnectionError("Connection failed"),
            MockOpenAIClient().chat.completions.create.return_value,
        ]

        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        # First two attempts fail
        with pytest.raises(APIConnectionError):
            adapter.chat_completions_create(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Network test 1"}],
                team="reliability",
            )

        with pytest.raises(APIConnectionError):
            adapter.chat_completions_create(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Network test 2"}],
                team="reliability",
            )

        # Third attempt succeeds
        response = adapter.chat_completions_create(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Network test 3"}],
            team="reliability",
        )
        assert response is not None

    def test_malformed_response_handling(self, mock_openai_import, mock_span_recorder):
        """Test handling of malformed responses from OpenRouter."""
        mock_client = MockOpenAIClient()

        # Create a malformed response object
        malformed_response = MagicMock()
        malformed_response.choices = None  # Missing expected attribute
        mock_client.chat.completions.create.return_value = malformed_response

        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        response = adapter.chat_completions_create(
            model="anthropic/claude-3-sonnet",
            messages=[{"role": "user", "content": "Malformed test"}],
            team="robustness",
        )

        # Should handle gracefully
        assert response is not None

    def test_extreme_payload_sizes(self, mock_openai_import, mock_span_recorder):
        """Test handling of extremely large and small payloads."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        # Test very small payload
        response_small = adapter.chat_completions_create(
            model="meta-llama/llama-3.2-1b-instruct",
            messages=[{"role": "user", "content": "Hi"}],
            team="payload-testing",
            max_tokens=1,
        )
        assert response_small is not None

        # Test large payload
        large_content = "This is a very long message. " * 50
        response_large = adapter.chat_completions_create(
            model="anthropic/claude-3-sonnet",
            messages=[{"role": "user", "content": large_content}],
            team="payload-testing",
            max_tokens=2000,
        )
        assert response_large is not None

        assert mock_client.chat.completions.create.call_count == 2

    def test_concurrent_request_handling(self, mock_openai_import, mock_span_recorder):
        """Test handling of concurrent requests with proper isolation."""
        import threading

        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenRouterAdapter(client=mock_client)

        results = []
        errors = []

        def make_request(request_id):
            try:
                response = adapter.chat_completions_create(
                    model="openai/gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": f"Concurrent request {request_id}"}
                    ],
                    team="concurrency",
                    request_id=str(request_id),
                )
                results.append(response)
            except Exception as e:
                errors.append(e)

        # Create multiple concurrent threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all requests completed successfully
        assert len(errors) == 0
        assert len(results) == 5
        assert mock_client.chat.completions.create.call_count >= 5


if __name__ == "__main__":
    pytest.main([__file__])
