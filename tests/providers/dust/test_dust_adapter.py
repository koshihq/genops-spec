"""Test suite for Dust provider adapter."""

from unittest.mock import Mock, patch

import pytest
import requests

from genops.providers.dust import GenOpsDustAdapter, auto_instrument, instrument_dust

# Constants to avoid CodeQL false positives
CONVERSATION_VISIBILITY_RESTRICTED = "private"


class TestGenOpsDustAdapter:
    """Test cases for GenOpsDustAdapter."""

    def test_adapter_initialization(self):
        """Test adapter initialization with valid parameters."""
        with patch("requests.Session") as mock_session:
            adapter = GenOpsDustAdapter(
                api_key="test-api-key",
                workspace_id="test-workspace",
                base_url="https://test.dust.tt",
            )

            assert adapter.api_key == "test-api-key"
            assert adapter.workspace_id == "test-workspace"
            assert adapter.base_url == "https://test.dust.tt"
            assert adapter.telemetry is not None

            # Verify session headers are set correctly
            mock_session.assert_called_once()
            session_instance = mock_session.return_value
            session_instance.headers.update.assert_called_once_with(
                {
                    "Authorization": "Bearer test-api-key",
                    "Content-Type": "application/json",
                }
            )

    def test_adapter_initialization_without_requests(self):
        """Test adapter initialization when requests is not available."""
        with patch("genops.providers.dust.HAS_REQUESTS", False):
            with pytest.raises(ImportError, match="requests package not found"):
                GenOpsDustAdapter(api_key="test-key", workspace_id="test-workspace")

    def test_extract_attributes(self):
        """Test attribute extraction from kwargs."""
        with patch("requests.Session"):
            adapter = GenOpsDustAdapter(
                api_key="test-key", workspace_id="test-workspace"
            )

            kwargs = {
                "team": "ai-team",
                "project": "test-project",
                "customer_id": "cust-123",
                "temperature": 0.7,
                "stream": True,
                "other_param": "value",
            }

            governance_attrs, request_attrs, api_kwargs = adapter._extract_attributes(
                kwargs
            )

            assert governance_attrs == {
                "team": "ai-team",
                "project": "test-project",
                "customer_id": "cust-123",
            }

            assert request_attrs == {"stream": True}

            assert api_kwargs == {
                "temperature": 0.7,
                "stream": True,
                "other_param": "value",
            }

    def test_make_request_success(self):
        """Test successful HTTP request."""
        with patch("requests.Session") as mock_session:
            # Setup mock response
            mock_response = Mock()
            mock_response.json.return_value = {"success": True}
            mock_session.return_value.request.return_value = mock_response

            adapter = GenOpsDustAdapter(
                api_key="test-key", workspace_id="test-workspace"
            )

            result = adapter._make_request("POST", "conversations", {"title": "test"})

            assert result == {"success": True}

            # Verify request was made correctly
            mock_session.return_value.request.assert_called_once_with(
                "POST",
                "https://dust.tt/api/v1/w/test-workspace/conversations",
                json={"title": "test"},
            )

    def test_make_request_error(self):
        """Test HTTP request error handling."""
        with patch("requests.Session") as mock_session:
            # Setup mock to raise exception
            mock_session.return_value.request.side_effect = requests.RequestException(
                "API Error"
            )

            adapter = GenOpsDustAdapter(
                api_key="test-key", workspace_id="test-workspace"
            )

            with pytest.raises(requests.RequestException):
                adapter._make_request("GET", "conversations")

    @patch("genops.providers.dust.GenOpsTelemetry")
    def test_create_conversation_success(self, mock_telemetry):
        """Test successful conversation creation."""
        with patch("requests.Session") as mock_session:
            # Setup mock response
            mock_response = Mock()
            mock_response.json.return_value = {
                "conversation": {"sId": "conv-123", "title": "Test Chat"}
            }
            mock_session.return_value.request.return_value = mock_response

            # Setup telemetry mock
            mock_span = Mock()
            mock_telemetry.return_value.trace_operation.return_value.__enter__.return_value = mock_span

            adapter = GenOpsDustAdapter(
                api_key="test-key", workspace_id="test-workspace"
            )

            result = adapter.create_conversation(
                title="Test Chat",
                visibility=CONVERSATION_VISIBILITY_RESTRICTED,
                team="ai-team",
                customer_id="cust-123",
            )

            assert result["conversation"]["sId"] == "conv-123"
            mock_span.set_attribute.assert_called_with("conversation_id", "conv-123")

    @patch("genops.providers.dust.GenOpsTelemetry")
    def test_send_message_success(self, mock_telemetry):
        """Test successful message sending."""
        with patch("requests.Session") as mock_session:
            # Setup mock response
            mock_response = Mock()
            mock_response.json.return_value = {
                "message": {"sId": "msg-456", "content": "Hello, world!"}
            }
            mock_session.return_value.request.return_value = mock_response

            # Setup telemetry mock
            mock_span = Mock()
            mock_telemetry.return_value.trace_operation.return_value.__enter__.return_value = mock_span

            adapter = GenOpsDustAdapter(
                api_key="test-key", workspace_id="test-workspace"
            )

            result = adapter.send_message(
                conversation_id="conv-123",
                content="Hello, world!",
                customer_id="cust-123",
            )

            assert result["message"]["sId"] == "msg-456"
            mock_span.set_attribute.assert_called_with("message_id", "msg-456")

    @patch("genops.providers.dust.GenOpsTelemetry")
    def test_run_agent_success(self, mock_telemetry):
        """Test successful agent execution."""
        with patch("requests.Session") as mock_session:
            # Setup mock response
            mock_response = Mock()
            mock_response.json.return_value = {
                "run": {
                    "sId": "run-789",
                    "status": "succeeded",
                    "results": [{"output": "Agent response"}],
                }
            }
            mock_session.return_value.request.return_value = mock_response

            # Setup telemetry mock
            mock_span = Mock()
            mock_telemetry.return_value.trace_operation.return_value.__enter__.return_value = mock_span

            adapter = GenOpsDustAdapter(
                api_key="test-key", workspace_id="test-workspace"
            )

            result = adapter.run_agent(
                agent_id="agent-abc", inputs={"query": "test query"}, team="ai-team"
            )

            assert result["run"]["sId"] == "run-789"
            assert result["run"]["status"] == "succeeded"
            mock_span.set_attribute.assert_any_call("run_id", "run-789")
            mock_span.set_attribute.assert_any_call("run_status", "succeeded")
            mock_span.set_attribute.assert_any_call("results_count", 1)

    @patch("genops.providers.dust.GenOpsTelemetry")
    def test_create_datasource_success(self, mock_telemetry):
        """Test successful datasource creation."""
        with patch("requests.Session") as mock_session:
            # Setup mock response
            mock_response = Mock()
            mock_response.json.return_value = {
                "data_source": {
                    "sId": "ds-123",
                    "name": "test-docs",
                    "description": "Test documentation",
                }
            }
            mock_session.return_value.request.return_value = mock_response

            # Setup telemetry mock
            mock_span = Mock()
            mock_telemetry.return_value.trace_operation.return_value.__enter__.return_value = mock_span

            adapter = GenOpsDustAdapter(
                api_key="test-key", workspace_id="test-workspace"
            )

            result = adapter.create_datasource(
                name="test-docs",
                description="Test documentation",
                visibility=CONVERSATION_VISIBILITY_RESTRICTED,
                project="test-project",
            )

            assert result["data_source"]["sId"] == "ds-123"
            mock_span.set_attribute.assert_called_with("datasource_id", "ds-123")

    @patch("genops.providers.dust.GenOpsTelemetry")
    def test_search_datasources_success(self, mock_telemetry):
        """Test successful datasource search."""
        with patch("requests.Session") as mock_session:
            # Setup mock response
            mock_response = Mock()
            mock_response.json.return_value = {
                "documents": [
                    {
                        "chunk": {
                            "text": "This is a test document about AI governance.",
                            "hash": "hash-123",
                        },
                        "score": 0.95,
                    },
                    {
                        "chunk": {
                            "text": "Another relevant document about cost tracking.",
                            "hash": "hash-456",
                        },
                        "score": 0.87,
                    },
                ]
            }
            mock_session.return_value.request.return_value = mock_response

            # Setup telemetry mock
            mock_span = Mock()
            mock_telemetry.return_value.trace_operation.return_value.__enter__.return_value = mock_span

            adapter = GenOpsDustAdapter(
                api_key="test-key", workspace_id="test-workspace"
            )

            result = adapter.search_datasources(
                query="AI governance",
                data_sources=["docs", "knowledge-base"],
                top_k=5,
                customer_id="cust-123",
            )

            assert len(result["documents"]) == 2
            mock_span.set_attribute.assert_any_call("documents_found", 2)
            # Should set estimated output tokens based on content
            mock_span.set_attribute.assert_any_call(
                "tokens_estimated_output", pytest.approx(20, rel=0.5)
            )

    @patch("genops.providers.dust.GenOpsTelemetry")
    def test_error_handling_with_telemetry(self, mock_telemetry):
        """Test error handling and telemetry recording."""
        with patch("requests.Session") as mock_session:
            # Setup mock to raise exception
            mock_session.return_value.request.side_effect = requests.RequestException(
                "API Error"
            )

            # Setup telemetry mock
            mock_span = Mock()
            mock_telemetry.return_value.trace_operation.return_value.__enter__.return_value = mock_span

            adapter = GenOpsDustAdapter(
                api_key="test-key", workspace_id="test-workspace"
            )

            with pytest.raises(requests.RequestException):
                adapter.create_conversation(title="Test")

            # Verify error was recorded in span
            mock_span.set_attribute.assert_any_call("error", True)
            mock_span.set_attribute.assert_any_call("error_message", "API Error")

    @patch("genops.core.context.get_effective_attributes")
    @patch("genops.providers.dust.GenOpsTelemetry")
    def test_context_integration(self, mock_telemetry, mock_get_effective_attributes):
        """Test integration with GenOps context system."""
        with patch("requests.Session") as mock_session:
            # Setup mock response
            mock_response = Mock()
            mock_response.json.return_value = {"conversation": {"sId": "conv-123"}}
            mock_session.return_value.request.return_value = mock_response

            # Setup context mock
            mock_get_effective_attributes.return_value = {
                "team": "context-team",
                "environment": "production",
                "cost_center": "ai-ops",
            }

            # Setup telemetry mock
            mock_span = Mock()
            mock_telemetry.return_value.trace_operation.return_value.__enter__.return_value = mock_span

            adapter = GenOpsDustAdapter(
                api_key="test-key", workspace_id="test-workspace"
            )

            adapter.create_conversation(
                title="Test",
                team="explicit-team",  # Should be merged with context
            )

            # Verify context was retrieved and used
            mock_get_effective_attributes.assert_called_once_with(team="explicit-team")


class TestInstrumentDust:
    """Test cases for instrument_dust convenience function."""

    def test_instrument_dust(self):
        """Test instrument_dust function creates adapter correctly."""
        with patch("requests.Session"):
            adapter = instrument_dust(
                api_key="test-key", workspace_id="test-workspace", team="test-team"
            )

            assert isinstance(adapter, GenOpsDustAdapter)
            assert adapter.api_key == "test-key"
            assert adapter.workspace_id == "test-workspace"


class TestAutoInstrument:
    """Test cases for auto_instrument function."""

    def test_auto_instrument(self):
        """Test auto_instrument function."""
        # This is mainly a placeholder function for Dust
        # since Dust doesn't have a standard Python SDK to wrap
        result = auto_instrument()
        assert result is None  # Should not raise exception


class TestAttributeExtraction:
    """Test cases for governance attribute handling."""

    def test_governance_attributes_separation(self):
        """Test that governance attributes are properly separated."""
        with patch("requests.Session"):
            adapter = GenOpsDustAdapter(
                api_key="test-key", workspace_id="test-workspace"
            )

            kwargs = {
                "team": "ai-team",
                "project": "customer-support",
                "feature": "conversation",
                "customer_id": "cust-123",
                "environment": "production",
                "cost_center": "support-ops",
                "user_id": "user-456",
                "stream": True,
                "blocking": False,
                "title": "Test Conversation",
            }

            governance_attrs, request_attrs, api_kwargs = adapter._extract_attributes(
                kwargs
            )

            expected_governance = {
                "team",
                "project",
                "feature",
                "customer_id",
                "environment",
                "cost_center",
                "user_id",
            }

            assert set(governance_attrs.keys()) == expected_governance
            assert "stream" in request_attrs
            assert "blocking" in request_attrs
            assert "title" in api_kwargs
            assert "stream" in api_kwargs  # Request attrs kept in api_kwargs too

    def test_empty_attributes(self):
        """Test handling of empty attribute dictionaries."""
        with patch("requests.Session"):
            adapter = GenOpsDustAdapter(
                api_key="test-key", workspace_id="test-workspace"
            )

            governance_attrs, request_attrs, api_kwargs = adapter._extract_attributes(
                {}
            )

            assert governance_attrs == {}
            assert request_attrs == {}
            assert api_kwargs == {}


class TestTelemetryAttributes:
    """Test cases for telemetry attribute generation."""

    @patch("genops.providers.dust.GenOpsTelemetry")
    def test_conversation_telemetry_attributes(self, mock_telemetry):
        """Test telemetry attributes for conversation creation."""
        with patch("requests.Session") as mock_session:
            mock_response = Mock()
            mock_response.json.return_value = {"conversation": {"sId": "conv-123"}}
            mock_session.return_value.request.return_value = mock_response

            # Capture the trace_operation call
            mock_span = Mock()
            trace_operation_mock = mock_telemetry.return_value.trace_operation
            trace_operation_mock.return_value.__enter__.return_value = mock_span

            adapter = GenOpsDustAdapter(
                api_key="test-key", workspace_id="test-workspace"
            )

            adapter.create_conversation(
                title="Test Chat",
                visibility="workspace",
                team="ai-team",
                project="customer-support",
            )

            # Verify trace_operation was called with correct attributes
            call_args = trace_operation_mock.call_args
            operation_name = call_args[0][0]
            attributes = call_args[1]

            assert operation_name == "dust.conversation.create"
            assert attributes["operation_type"] == "ai.conversation"
            assert attributes["provider"] == "dust"
            assert attributes["conversation_title"] == "Test Chat"
            assert attributes["visibility"] == "workspace"
            assert attributes["workspace_id"] == "test-workspace"

    @patch("genops.providers.dust.GenOpsTelemetry")
    def test_message_telemetry_attributes(self, mock_telemetry):
        """Test telemetry attributes for message sending."""
        with patch("requests.Session") as mock_session:
            mock_response = Mock()
            mock_response.json.return_value = {"message": {"sId": "msg-123"}}
            mock_session.return_value.request.return_value = mock_response

            # Capture the trace_operation call
            mock_span = Mock()
            trace_operation_mock = mock_telemetry.return_value.trace_operation
            trace_operation_mock.return_value.__enter__.return_value = mock_span

            adapter = GenOpsDustAdapter(
                api_key="test-key", workspace_id="test-workspace"
            )

            message_content = "This is a test message for token estimation."
            adapter.send_message(
                conversation_id="conv-123",
                content=message_content,
                customer_id="cust-456",
            )

            # Verify trace_operation was called with correct attributes
            call_args = trace_operation_mock.call_args
            operation_name = call_args[0][0]
            attributes = call_args[1]

            assert operation_name == "dust.message.send"
            assert attributes["operation_type"] == "ai.message"
            assert attributes["provider"] == "dust"
            assert attributes["conversation_id"] == "conv-123"
            assert attributes["workspace_id"] == "test-workspace"
            assert attributes["message_length"] == len(message_content)
            # Token estimation should be roughly words * 1.3
            expected_tokens = int(len(message_content.split()) * 1.3)
            assert attributes["tokens_estimated_input"] == expected_tokens
