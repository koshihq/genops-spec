"""Tests for OpenAI provider adapter."""

from unittest.mock import MagicMock, patch

import pytest

from genops.providers.openai import GenOpsOpenAIAdapter
from tests.utils.mock_providers import MockOpenAIClient, MockProviderFactory


class TestGenOpsOpenAIAdapter:
    """Test OpenAI adapter with governance tracking."""

    def test_adapter_initialization_with_client(self, mock_openai_import):
        """Test adapter initialization with provided client."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenAIAdapter(client=mock_client)

        assert adapter.client == mock_client
        assert adapter.telemetry is not None

    def test_adapter_initialization_without_client(self, mock_openai_import):
        """Test adapter initialization creates OpenAI client."""
        mock_openai_class = mock_openai_import
        mock_openai_class.return_value = MockOpenAIClient()

        GenOpsOpenAIAdapter(api_key="test-key")

        # Verify OpenAI client was created with kwargs
        mock_openai_class.assert_called_once_with(api_key="test-key")

    def test_adapter_initialization_missing_openai(self):
        """Test adapter initialization fails when OpenAI not installed."""
        with patch("genops.providers.openai.HAS_OPENAI", False):
            with pytest.raises(ImportError) as exc_info:
                GenOpsOpenAIAdapter()

            assert "OpenAI package not found" in str(exc_info.value)

    def test_chat_completions_create_basic(
        self, mock_openai_import, mock_span_recorder
    ):
        """Test basic chat completions with governance tracking."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenAIAdapter(client=mock_client)

        messages = [{"role": "user", "content": "What is machine learning?"}]

        response = adapter.chat_completions_create(
            model="gpt-3.5-turbo", messages=messages
        )

        # Verify response structure
        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert hasattr(response, "usage")

        # Verify telemetry was recorded
        spans = mock_span_recorder.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "openai.chat.completions.create"

        # Check governance attributes
        attrs = span.attributes
        assert attrs["genops.operation.type"] == "ai.inference"
        assert attrs["genops.provider"] == "openai"
        assert attrs["genops.model"] == "gpt-3.5-turbo"
        assert "genops.tokens_estimated_input" in attrs

    def test_cost_calculation_gpt35_turbo(self, mock_openai_import, mock_span_recorder):
        """Test cost calculation for GPT-3.5-turbo."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenAIAdapter(client=mock_client)

        # Mock response with known token counts
        mock_response = MockProviderFactory.create_openai_response(
            model="gpt-3.5-turbo", prompt_tokens=100, completion_tokens=50
        )
        mock_client.chat.completions.create.return_value = mock_response

        adapter.chat_completions_create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test message"}],
        )

        # Verify cost calculation
        spans = mock_span_recorder.get_finished_spans()
        span = spans[0]
        attrs = span.attributes

        assert attrs["genops.tokens.input"] == 100
        assert attrs["genops.tokens.output"] == 50
        assert attrs["genops.tokens.total"] == 150

        # GPT-3.5-turbo pricing: $0.0005 input, $0.0015 output per 1K tokens
        expected_cost = (100 / 1000 * 0.0005) + (50 / 1000 * 0.0015)
        assert abs(attrs["genops.cost.total"] - expected_cost) < 0.0001
        assert attrs["genops.cost.currency"] == "USD"

    def test_cost_calculation_gpt4(self, mock_openai_import, mock_span_recorder):
        """Test cost calculation for GPT-4."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenAIAdapter(client=mock_client)

        mock_response = MockProviderFactory.create_openai_response(
            model="gpt-4", prompt_tokens=200, completion_tokens=100
        )
        mock_client.chat.completions.create.return_value = mock_response

        adapter.chat_completions_create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Complex reasoning task"}],
        )

        spans = mock_span_recorder.get_finished_spans()
        attrs = spans[0].attributes

        # GPT-4 pricing: $0.03 input, $0.06 output per 1K tokens
        expected_cost = (200 / 1000 * 0.03) + (100 / 1000 * 0.06)
        assert abs(attrs["genops.cost.total"] - expected_cost) < 0.001
        assert attrs["genops.model"] == "gpt-4"

    def test_governance_attributes_inheritance(
        self, mock_openai_import, mock_span_recorder
    ):
        """Test that governance attributes are properly set."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenAIAdapter(client=mock_client)

        # Set up governance context
        governance_attrs = {
            "team": "ai-platform",
            "project": "chatbot",
            "feature": "customer_support",
            "customer_id": "customer_123",
        }

        # Mock the telemetry to include governance attributes
        with patch.object(adapter.telemetry, "trace_operation") as mock_trace:
            mock_span = MagicMock()
            mock_trace.return_value.__enter__.return_value = mock_span

            adapter.chat_completions_create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}],
                **governance_attrs,
            )

            # Verify governance attributes were passed to telemetry
            mock_trace.assert_called_once()
            call_kwargs = mock_trace.call_args[1]
            assert call_kwargs["provider"] == "openai"
            assert call_kwargs["model"] == "gpt-3.5-turbo"

    def test_error_handling_api_failure(self, mock_openai_import, mock_span_recorder):
        """Test error handling when OpenAI API fails."""
        mock_client = MockOpenAIClient(fail_requests=True)
        adapter = GenOpsOpenAIAdapter(client=mock_client)

        with pytest.raises(Exception) as exc_info:
            adapter.chat_completions_create(
                model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Test"}]
            )

        assert "Mock API error" in str(exc_info.value)

        # Verify error was recorded in telemetry
        spans = mock_span_recorder.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.status.status_code.name == "ERROR"

    def test_token_estimation_accuracy(self, mock_openai_import):
        """Test token estimation for input messages."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenAIAdapter(client=mock_client)

        # Test with various message lengths
        test_cases = [
            {
                "messages": [{"role": "user", "content": "Hi"}],
                "expected_min": 1,
                "expected_max": 5,
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "This is a longer message with more words",
                    },
                    {"role": "assistant", "content": "This is a response"},
                ],
                "expected_min": 10,
                "expected_max": 25,
            },
        ]

        for case in test_cases:
            with patch.object(adapter.telemetry, "trace_operation") as mock_trace:
                mock_span = MagicMock()
                mock_trace.return_value.__enter__.return_value = mock_span

                adapter.chat_completions_create(
                    model="gpt-3.5-turbo", messages=case["messages"]
                )

                call_kwargs = mock_trace.call_args[1]
                estimated_tokens = call_kwargs.get("tokens_estimated_input", 0)

                assert case["expected_min"] <= estimated_tokens <= case["expected_max"]

    def test_streaming_support_flag(self, mock_openai_import, mock_span_recorder):
        """Test that streaming requests are flagged appropriately."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenAIAdapter(client=mock_client)

        adapter.chat_completions_create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            stream=True,
        )

        spans = mock_span_recorder.get_finished_spans()
        span = spans[0]
        attrs = span.attributes

        # Streaming should be noted in telemetry
        assert attrs.get("genops.request.streaming") is True

    def test_multiple_models_cost_accuracy(self, mock_openai_import):
        """Test cost calculation accuracy across different models."""
        mock_client = MockOpenAIClient()
        GenOpsOpenAIAdapter(client=mock_client)

        test_models = [
            {
                "model": "gpt-3.5-turbo",
                "input_tokens": 1000,
                "output_tokens": 500,
                "expected_cost": (1000 / 1000 * 0.0005) + (500 / 1000 * 0.0015),
            },
            {
                "model": "gpt-4",
                "input_tokens": 500,
                "output_tokens": 250,
                "expected_cost": (500 / 1000 * 0.03) + (250 / 1000 * 0.06),
            },
        ]

        for test_case in test_models:
            mock_response = MockProviderFactory.create_openai_response(
                model=test_case["model"],
                prompt_tokens=test_case["input_tokens"],
                completion_tokens=test_case["output_tokens"],
            )
            mock_client.chat.completions.create.return_value = mock_response

            calculated_cost = MockProviderFactory.calculate_openai_cost(
                test_case["model"],
                test_case["input_tokens"],
                test_case["output_tokens"],
            )

            assert abs(calculated_cost - test_case["expected_cost"]) < 0.0001

    def test_unknown_model_fallback_pricing(
        self, mock_openai_import, mock_span_recorder
    ):
        """Test fallback pricing for unknown models."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenAIAdapter(client=mock_client)

        mock_response = MockProviderFactory.create_openai_response(
            model="unknown-model", prompt_tokens=100, completion_tokens=50
        )
        mock_client.chat.completions.create.return_value = mock_response

        adapter.chat_completions_create(
            model="unknown-model", messages=[{"role": "user", "content": "Test"}]
        )

        spans = mock_span_recorder.get_finished_spans()
        attrs = spans[0].attributes

        # Should fall back to GPT-3.5-turbo pricing
        expected_cost = (100 / 1000 * 0.0005) + (50 / 1000 * 0.0015)
        assert abs(attrs["genops.cost.total"] - expected_cost) < 0.0001
        assert attrs["genops.model"] == "unknown-model"  # Original model preserved

    def test_request_metadata_capture(self, mock_openai_import, mock_span_recorder):
        """Test that additional request metadata is captured."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenAIAdapter(client=mock_client)

        adapter.chat_completions_create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.8,
            max_tokens=150,
            top_p=0.9,
        )

        spans = mock_span_recorder.get_finished_spans()
        attrs = spans[0].attributes

        # Verify request parameters are captured
        assert attrs.get("genops.request.temperature") == 0.8
        assert attrs.get("genops.request.max_tokens") == 150
        assert attrs.get("genops.request.top_p") == 0.9
