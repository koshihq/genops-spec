"""Tests for Anthropic provider adapter."""

import pytest
from unittest.mock import MagicMock, patch
from tests.utils.mock_providers import MockAnthropicClient, MockProviderFactory

from genops.providers.anthropic import GenOpsAnthropicAdapter


class TestGenOpsAnthropicAdapter:
    """Test Anthropic adapter with governance tracking."""
    
    def test_adapter_initialization_with_client(self, mock_anthropic_import):
        """Test adapter initialization with provided client."""
        mock_client = MockAnthropicClient()
        adapter = GenOpsAnthropicAdapter(client=mock_client)
        
        assert adapter.client == mock_client
        assert adapter.telemetry is not None
    
    def test_adapter_initialization_without_client(self, mock_anthropic_import):
        """Test adapter initialization creates Anthropic client."""
        mock_anthropic_class = mock_anthropic_import
        mock_anthropic_class.return_value = MockAnthropicClient()
        
        adapter = GenOpsAnthropicAdapter(api_key="test-key")
        
        # Verify Anthropic client was created with kwargs
        mock_anthropic_class.assert_called_once_with(api_key="test-key")
    
    def test_adapter_initialization_missing_anthropic(self):
        """Test adapter initialization fails when Anthropic not installed."""
        with patch('genops.providers.anthropic.HAS_ANTHROPIC', False):
            with pytest.raises(ImportError) as exc_info:
                GenOpsAnthropicAdapter()
            
            assert "Anthropic package not found" in str(exc_info.value)
    
    def test_messages_create_basic(self, mock_anthropic_import, mock_span_recorder):
        """Test basic messages create with governance tracking."""
        mock_client = MockAnthropicClient()
        adapter = GenOpsAnthropicAdapter(client=mock_client)
        
        messages = [
            {"role": "user", "content": "What is machine learning?"}
        ]
        
        response = adapter.messages_create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=messages
        )
        
        # Verify response structure
        assert response is not None
        assert hasattr(response, 'content')
        assert len(response.content) > 0
        assert hasattr(response, 'usage')
        
        # Verify telemetry was recorded
        spans = mock_span_recorder.get_finished_spans()
        assert len(spans) == 1
        
        span = spans[0]
        assert span.name == "anthropic.messages.create"
        
        # Check governance attributes
        attrs = span.attributes
        assert attrs["genops.operation.type"] == "ai.inference"
        assert attrs["genops.provider"] == "anthropic"
        assert attrs["genops.model"] == "claude-3-sonnet-20240229"
        assert attrs["genops.request.max_tokens"] == 1024
    
    def test_cost_calculation_claude3_sonnet(self, mock_anthropic_import, mock_span_recorder):
        """Test cost calculation for Claude-3 Sonnet."""
        mock_client = MockAnthropicClient()
        adapter = GenOpsAnthropicAdapter(client=mock_client)
        
        # Mock response with known token counts
        mock_response = MockProviderFactory.create_anthropic_response(
            model="claude-3-sonnet-20240229",
            input_tokens=120,
            output_tokens=80
        )
        mock_client.messages.create.return_value = mock_response
        
        response = adapter.messages_create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Test message"}]
        )
        
        # Verify cost calculation
        spans = mock_span_recorder.get_finished_spans()
        span = spans[0]
        attrs = span.attributes
        
        assert attrs["genops.tokens.input"] == 120
        assert attrs["genops.tokens.output"] == 80
        assert attrs["genops.tokens.total"] == 200
        
        # Claude-3 Sonnet pricing: $0.003 input, $0.015 output per 1K tokens
        expected_cost = (120 / 1000 * 0.003) + (80 / 1000 * 0.015)
        assert abs(attrs["genops.cost.total"] - expected_cost) < 0.0001
        assert attrs["genops.cost.currency"] == "USD"
    
    def test_cost_calculation_claude3_opus(self, mock_anthropic_import, mock_span_recorder):
        """Test cost calculation for Claude-3 Opus."""
        mock_client = MockAnthropicClient()
        adapter = GenOpsAnthropicAdapter(client=mock_client)
        
        mock_response = MockProviderFactory.create_anthropic_response(
            model="claude-3-opus-20240229",
            input_tokens=100,
            output_tokens=150
        )
        mock_client.messages.create.return_value = mock_response
        
        response = adapter.messages_create(
            model="claude-3-opus-20240229",
            max_tokens=2048,
            messages=[{"role": "user", "content": "Complex analysis task"}]
        )
        
        spans = mock_span_recorder.get_finished_spans()
        attrs = spans[0].attributes
        
        # Claude-3 Opus pricing: $0.015 input, $0.075 output per 1K tokens
        expected_cost = (100 / 1000 * 0.015) + (150 / 1000 * 0.075)
        assert abs(attrs["genops.cost.total"] - expected_cost) < 0.001
        assert attrs["genops.model"] == "claude-3-opus-20240229"
    
    def test_cost_calculation_claude3_haiku(self, mock_anthropic_import, mock_span_recorder):
        """Test cost calculation for Claude-3 Haiku."""
        mock_client = MockAnthropicClient()
        adapter = GenOpsAnthropicAdapter(client=mock_client)
        
        mock_response = MockProviderFactory.create_anthropic_response(
            model="claude-3-haiku-20240307",
            input_tokens=200,
            output_tokens=50
        )
        mock_client.messages.create.return_value = mock_response
        
        response = adapter.messages_create(
            model="claude-3-haiku-20240307",
            max_tokens=512,
            messages=[{"role": "user", "content": "Quick task"}]
        )
        
        spans = mock_span_recorder.get_finished_spans()
        attrs = spans[0].attributes
        
        # Claude-3 Haiku pricing: $0.00025 input, $0.00125 output per 1K tokens
        expected_cost = (200 / 1000 * 0.00025) + (50 / 1000 * 0.00125)
        assert abs(attrs["genops.cost.total"] - expected_cost) < 0.00001
        assert attrs["genops.model"] == "claude-3-haiku-20240307"
    
    def test_governance_attributes_inheritance(self, mock_anthropic_import, mock_span_recorder):
        """Test that governance attributes are properly set."""
        mock_client = MockAnthropicClient()
        adapter = GenOpsAnthropicAdapter(client=mock_client)
        
        # Set up governance context
        governance_attrs = {
            "team": "research-team",
            "project": "document-analysis",
            "feature": "pdf_extraction",
            "customer_id": "enterprise_123"
        }
        
        # Mock the telemetry to include governance attributes
        with patch.object(adapter.telemetry, 'trace_operation') as mock_trace:
            mock_span = MagicMock()
            mock_trace.return_value.__enter__.return_value = mock_span
            
            response = adapter.messages_create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Test"}],
                **governance_attrs
            )
            
            # Verify governance attributes were passed to telemetry
            mock_trace.assert_called_once()
            call_kwargs = mock_trace.call_args[1]
            assert call_kwargs["provider"] == "anthropic"
            assert call_kwargs["model"] == "claude-3-sonnet-20240229"
    
    def test_error_handling_api_failure(self, mock_anthropic_import, mock_span_recorder):
        """Test error handling when Anthropic API fails."""
        mock_client = MockAnthropicClient(fail_requests=True)
        adapter = GenOpsAnthropicAdapter(client=mock_client)
        
        with pytest.raises(Exception) as exc_info:
            adapter.messages_create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Test"}]
            )
        
        assert "Mock API error" in str(exc_info.value)
        
        # Verify error was recorded in telemetry
        spans = mock_span_recorder.get_finished_spans()
        assert len(spans) == 1
        
        span = spans[0]
        assert span.status.status_code.name == "ERROR"
    
    def test_system_message_handling(self, mock_anthropic_import, mock_span_recorder):
        """Test handling of system messages in Claude API."""
        mock_client = MockAnthropicClient()
        adapter = GenOpsAnthropicAdapter(client=mock_client)
        
        response = adapter.messages_create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            system="You are a helpful assistant that provides concise answers.",
            messages=[{"role": "user", "content": "What is AI?"}]
        )
        
        spans = mock_span_recorder.get_finished_spans()
        span = spans[0]
        attrs = span.attributes
        
        # System message should be captured
        assert attrs.get("genops.request.system") == "You are a helpful assistant that provides concise answers."
    
    def test_streaming_support_flag(self, mock_anthropic_import, mock_span_recorder):
        """Test that streaming requests are flagged appropriately."""
        mock_client = MockAnthropicClient()
        adapter = GenOpsAnthropicAdapter(client=mock_client)
        
        response = adapter.messages_create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Test"}],
            stream=True
        )
        
        spans = mock_span_recorder.get_finished_spans()
        span = spans[0]
        attrs = span.attributes
        
        # Streaming should be noted in telemetry
        assert attrs.get("genops.request.streaming") is True
    
    def test_temperature_and_parameters_capture(self, mock_anthropic_import, mock_span_recorder):
        """Test that request parameters are captured in telemetry."""
        mock_client = MockAnthropicClient()
        adapter = GenOpsAnthropicAdapter(client=mock_client)
        
        response = adapter.messages_create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            messages=[{"role": "user", "content": "Test"}]
        )
        
        spans = mock_span_recorder.get_finished_spans()
        attrs = spans[0].attributes
        
        # Verify request parameters are captured
        assert attrs.get("genops.request.temperature") == 0.7
        assert attrs.get("genops.request.top_p") == 0.9
        assert attrs.get("genops.request.top_k") == 40
        assert attrs.get("genops.request.max_tokens") == 1024
    
    def test_unknown_model_fallback_pricing(self, mock_anthropic_import, mock_span_recorder):
        """Test fallback pricing for unknown Claude models."""
        mock_client = MockAnthropicClient()
        adapter = GenOpsAnthropicAdapter(client=mock_client)
        
        mock_response = MockProviderFactory.create_anthropic_response(
            model="claude-unknown-model",
            input_tokens=100,
            output_tokens=50
        )
        mock_client.messages.create.return_value = mock_response
        
        response = adapter.messages_create(
            model="claude-unknown-model",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Test"}]
        )
        
        spans = mock_span_recorder.get_finished_spans()
        attrs = spans[0].attributes
        
        # Should fall back to Claude-3 Sonnet pricing
        expected_cost = (100/1000 * 0.003) + (50/1000 * 0.015)
        assert abs(attrs["genops.cost.total"] - expected_cost) < 0.0001
        assert attrs["genops.model"] == "claude-unknown-model"  # Original model preserved
    
    def test_multiple_content_blocks(self, mock_anthropic_import, mock_span_recorder):
        """Test handling of multiple content blocks in response."""
        mock_client = MockAnthropicClient()
        adapter = GenOpsAnthropicAdapter(client=mock_client)
        
        # Mock response with multiple content blocks
        mock_response = MockProviderFactory.create_anthropic_response()
        # Add additional content blocks
        additional_content = MagicMock()
        additional_content.type = "text"
        additional_content.text = "Additional response content"
        mock_response.content.append(additional_content)
        
        mock_client.messages.create.return_value = mock_response
        
        response = adapter.messages_create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Complex task requiring multiple parts"}]
        )
        
        # Should handle multiple content blocks gracefully
        assert len(response.content) == 2
        
        spans = mock_span_recorder.get_finished_spans()
        span = spans[0]
        attrs = span.attributes
        
        # Should capture total response length
        assert "genops.response.content_blocks" in attrs
        assert attrs["genops.response.content_blocks"] == 2
    
    def test_claude_instant_legacy_model(self, mock_anthropic_import, mock_span_recorder):
        """Test support for legacy Claude Instant model."""
        mock_client = MockAnthropicClient()
        adapter = GenOpsAnthropicAdapter(client=mock_client)
        
        mock_response = MockProviderFactory.create_anthropic_response(
            model="claude-instant-1.2",
            input_tokens=150,
            output_tokens=75
        )
        mock_client.messages.create.return_value = mock_response
        
        response = adapter.messages_create(
            model="claude-instant-1.2",
            max_tokens=512,
            messages=[{"role": "user", "content": "Quick question"}]
        )
        
        spans = mock_span_recorder.get_finished_spans()
        attrs = spans[0].attributes
        
        # Claude Instant pricing: $0.00163 input, $0.00551 output per 1K tokens
        expected_cost = (150/1000 * 0.00163) + (75/1000 * 0.00551)
        assert abs(attrs["genops.cost.total"] - expected_cost) < 0.0001
        assert attrs["genops.model"] == "claude-instant-1.2"
    
    def test_large_context_handling(self, mock_anthropic_import, mock_span_recorder):
        """Test handling of large context messages."""
        mock_client = MockAnthropicClient()
        adapter = GenOpsAnthropicAdapter(client=mock_client)
        
        # Create a large message
        large_content = "This is a test message. " * 1000  # ~5000 characters
        messages = [{"role": "user", "content": large_content}]
        
        mock_response = MockProviderFactory.create_anthropic_response(
            input_tokens=2000,  # Large input token count
            output_tokens=500
        )
        mock_client.messages.create.return_value = mock_response
        
        response = adapter.messages_create(
            model="claude-3-sonnet-20240229",
            max_tokens=2048,
            messages=messages
        )
        
        spans = mock_span_recorder.get_finished_spans()
        attrs = spans[0].attributes
        
        assert attrs["genops.tokens.input"] == 2000
        assert attrs["genops.tokens.output"] == 500
        # Large context should still be handled properly
        assert attrs["genops.cost.total"] > 0