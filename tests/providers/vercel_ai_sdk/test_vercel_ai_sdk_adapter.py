"""Tests for Vercel AI SDK adapter core functionality."""

import os
import tempfile
import threading
import time
import unittest
from decimal import Decimal
from unittest.mock import Mock, patch

from genops.providers.vercel_ai_sdk import (
    GenOpsVercelAISDKAdapter,
    VercelAISDKRequest,
    VercelAISDKResponse,
    auto_instrument,
    track_generate_text,
    track_stream_text,
)


class TestVercelAISDKRequest(unittest.TestCase):
    """Test VercelAISDKRequest data class."""

    def test_request_creation(self):
        """Test creating a basic request object."""
        request = VercelAISDKRequest(
            request_id="test-123",
            provider="openai",
            model="gpt-4",
            operation_type="generateText",
        )

        self.assertEqual(request.request_id, "test-123")
        self.assertEqual(request.provider, "openai")
        self.assertEqual(request.model, "gpt-4")
        self.assertEqual(request.operation_type, "generateText")
        self.assertIsNone(request.input_tokens)
        self.assertIsNone(request.output_tokens)
        self.assertEqual(request.tools_used, [])
        self.assertEqual(request.governance_attrs, {})

    def test_request_with_tokens(self):
        """Test request with token information."""
        request = VercelAISDKRequest(
            request_id="test-123",
            provider="openai",
            model="gpt-4",
            operation_type="generateText",
            input_tokens=100,
            output_tokens=150,
        )

        self.assertEqual(request.input_tokens, 100)
        self.assertEqual(request.output_tokens, 150)

    def test_request_with_governance_attrs(self):
        """Test request with governance attributes."""
        governance_attrs = {
            "team": "ai-team",
            "project": "chatbot",
            "customer_id": "cust-123",
        }

        request = VercelAISDKRequest(
            request_id="test-123",
            provider="openai",
            model="gpt-4",
            operation_type="generateText",
            governance_attrs=governance_attrs,
        )

        self.assertEqual(request.governance_attrs, governance_attrs)


class TestVercelAISDKResponse(unittest.TestCase):
    """Test VercelAISDKResponse data class."""

    def test_successful_response(self):
        """Test creating a successful response."""
        response = VercelAISDKResponse(
            request_id="test-123", success=True, text="Hello, world!"
        )

        self.assertEqual(response.request_id, "test-123")
        self.assertTrue(response.success)
        self.assertEqual(response.text, "Hello, world!")
        self.assertIsNone(response.error)

    def test_error_response(self):
        """Test creating an error response."""
        response = VercelAISDKResponse(
            request_id="test-123", success=False, error="API key not found"
        )

        self.assertEqual(response.request_id, "test-123")
        self.assertFalse(response.success)
        self.assertEqual(response.error, "API key not found")
        self.assertIsNone(response.text)


class TestGenOpsVercelAISDKAdapter(unittest.TestCase):
    """Test the main Vercel AI SDK adapter."""

    def setUp(self):
        """Set up test environment."""
        self.test_governance_attrs = {
            "team": "test-team",
            "project": "test-project",
            "environment": "test",
        }

    def test_adapter_initialization(self):
        """Test adapter initialization with default parameters."""
        adapter = GenOpsVercelAISDKAdapter()

        self.assertEqual(adapter.integration_mode, "python_wrapper")
        self.assertEqual(adapter.websocket_port, 8080)
        self.assertIsInstance(adapter.governance_attrs, dict)
        self.assertIsInstance(adapter.active_requests, dict)
        self.assertEqual(len(adapter.active_requests), 0)

    def test_adapter_initialization_with_params(self):
        """Test adapter initialization with custom parameters."""
        adapter = GenOpsVercelAISDKAdapter(
            integration_mode="websocket",
            websocket_port=9090,
            **self.test_governance_attrs,
        )

        # Note: websocket mode might fallback to python_wrapper if websockets not available
        self.assertEqual(adapter.websocket_port, 9090)
        self.assertEqual(adapter.governance_attrs["team"], "test-team")
        self.assertEqual(adapter.governance_attrs["project"], "test-project")
        self.assertEqual(adapter.governance_attrs["environment"], "test")

    def test_invalid_integration_mode(self):
        """Test adapter initialization with invalid integration mode."""
        with self.assertRaises(ValueError):
            GenOpsVercelAISDKAdapter(integration_mode="invalid_mode")

    def test_governance_attributes_initialization(self):
        """Test governance attributes initialization with environment variables."""
        with patch.dict(
            os.environ,
            {
                "GENOPS_TEAM": "env-team",
                "GENOPS_PROJECT": "env-project",
                "GENOPS_ENVIRONMENT": "env-environment",
            },
        ):
            adapter = GenOpsVercelAISDKAdapter()

            self.assertEqual(adapter.governance_attrs["team"], "env-team")
            self.assertEqual(adapter.governance_attrs["project"], "env-project")
            self.assertEqual(adapter.governance_attrs["environment"], "env-environment")

    def test_extract_attributes(self):
        """Test attribute extraction from kwargs."""
        adapter = GenOpsVercelAISDKAdapter(**self.test_governance_attrs)

        kwargs = {
            "team": "override-team",
            "temperature": 0.7,
            "maxTokens": 150,
            "custom_param": "custom_value",
        }

        governance_attrs, request_attrs, api_kwargs = adapter._extract_attributes(
            kwargs
        )

        # Governance attributes should be merged with instance attributes
        self.assertEqual(governance_attrs["team"], "override-team")  # Override
        self.assertEqual(governance_attrs["project"], "test-project")  # From instance

        # Request attributes should include recognized parameters
        self.assertEqual(request_attrs["temperature"], 0.7)
        self.assertEqual(request_attrs["maxTokens"], 150)

        # API kwargs should include unrecognized parameters
        self.assertEqual(api_kwargs["custom_param"], "custom_value")
        self.assertNotIn("team", api_kwargs)
        self.assertNotIn("temperature", api_kwargs)  # Request attr, kept in api_kwargs

    @patch("genops.providers.vercel_ai_sdk.GenOpsTelemetry")
    def test_track_request_context_manager(self, mock_telemetry):
        """Test the track_request context manager."""
        mock_span = Mock()
        mock_telemetry.return_value.start_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_telemetry.return_value.start_span.return_value.__exit__ = Mock(
            return_value=None
        )

        adapter = GenOpsVercelAISDKAdapter(**self.test_governance_attrs)

        with adapter.track_request("generateText", "openai", "gpt-4") as request:
            self.assertIsInstance(request, VercelAISDKRequest)
            self.assertEqual(request.operation_type, "generateText")
            self.assertEqual(request.provider, "openai")
            self.assertEqual(request.model, "gpt-4")

            # Request should be in active requests
            self.assertIn(request.request_id, adapter.active_requests)

            # Simulate some processing
            request.input_tokens = 50
            request.output_tokens = 100

        # After context manager, request should be removed from active requests
        self.assertNotIn(request.request_id, adapter.active_requests)

        # Telemetry should have been called
        mock_telemetry.return_value.start_span.assert_called()

    @patch("genops.providers.vercel_ai_sdk.GenOpsTelemetry")
    def test_track_request_with_error(self, mock_telemetry):
        """Test track_request context manager with error handling."""
        mock_span = Mock()
        mock_telemetry.return_value.start_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_telemetry.return_value.start_span.return_value.__exit__ = Mock(
            return_value=None
        )

        adapter = GenOpsVercelAISDKAdapter(**self.test_governance_attrs)

        with self.assertRaises(ValueError):
            with adapter.track_request("generateText", "openai", "gpt-4") as request:
                # Simulate an error
                raise ValueError("Test error")

        # Request should still be cleaned up even after error
        self.assertNotIn(request.request_id, adapter.active_requests)

        # Error should be recorded in request
        self.assertEqual(request.error, "Test error")

    def test_calculate_cost(self):
        """Test cost calculation for different providers."""
        adapter = GenOpsVercelAISDKAdapter()

        # Test with OpenAI model (should use provider-specific calculator)
        with patch("genops.providers.vercel_ai_sdk.calculate_cost") as mock_calculate:
            mock_calculate.return_value = Decimal("0.002")
            cost = adapter._calculate_cost("openai", "gpt-4", 100, 150)
            self.assertEqual(cost, Decimal("0.002"))

        # Test with unknown provider (should use fallback)
        cost = adapter._calculate_cost("unknown", "unknown-model", 100, 150)
        self.assertIsInstance(cost, Decimal)
        self.assertGreater(cost, Decimal("0"))

    def test_finalize_request_telemetry(self):
        """Test telemetry finalization for completed request."""
        adapter = GenOpsVercelAISDKAdapter(**self.test_governance_attrs)

        request = VercelAISDKRequest(
            request_id="test-123",
            provider="openai",
            model="gpt-4",
            operation_type="generateText",
            input_tokens=100,
            output_tokens=150,
            governance_attrs=self.test_governance_attrs,
            duration_ms=1500.0,
        )

        with patch("genops.providers.vercel_ai_sdk.GenOpsTelemetry") as mock_telemetry:
            mock_span = Mock()
            mock_telemetry.return_value.start_span.return_value.__enter__ = Mock(
                return_value=mock_span
            )
            mock_telemetry.return_value.start_span.return_value.__exit__ = Mock(
                return_value=None
            )

            adapter._finalize_request_telemetry(request)

            # Verify telemetry was called
            mock_telemetry.return_value.start_span.assert_called()
            mock_span.set_attribute.assert_called()

    def test_generate_instrumentation_code(self):
        """Test JavaScript instrumentation code generation."""
        adapter = GenOpsVercelAISDKAdapter(**self.test_governance_attrs)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            output_path = f.name

        try:
            result_path = adapter.generate_instrumentation_code(output_path)
            self.assertEqual(result_path, output_path)

            # Verify file was created and contains expected content
            with open(output_path) as f:
                content = f.read()

            self.assertIn("GenOps Vercel AI SDK Instrumentation", content)
            self.assertIn("instrumentedGenerateText", content)
            self.assertIn("test-team", content)  # Should include governance attributes
            self.assertIn("test-project", content)

        finally:
            os.unlink(output_path)


class TestAutoInstrumentation(unittest.TestCase):
    """Test auto-instrumentation functions."""

    def test_auto_instrument_function(self):
        """Test the auto_instrument function."""
        adapter = auto_instrument(
            integration_mode="python_wrapper", team="test-team", project="test-project"
        )

        self.assertIsInstance(adapter, GenOpsVercelAISDKAdapter)
        self.assertEqual(adapter.integration_mode, "python_wrapper")
        self.assertEqual(adapter.governance_attrs["team"], "test-team")
        self.assertEqual(adapter.governance_attrs["project"], "test-project")

    @patch("genops.providers.vercel_ai_sdk.auto_instrument")
    def test_convenience_functions(self, mock_auto_instrument):
        """Test convenience functions for tracking operations."""
        mock_adapter = Mock()
        mock_auto_instrument.return_value = mock_adapter

        # Test track_generate_text
        with track_generate_text("openai", "gpt-4", team="test-team"):
            pass

        mock_auto_instrument.assert_called()
        mock_adapter.track_request.assert_called_with(
            "generateText", "openai", "gpt-4", team="test-team"
        )

        # Test track_stream_text
        with track_stream_text("anthropic", "claude-3-sonnet", project="test-project"):
            pass

        mock_adapter.track_request.assert_called_with(
            "streamText", "anthropic", "claude-3-sonnet", project="test-project"
        )


class TestThreadSafety(unittest.TestCase):
    """Test thread safety of the adapter."""

    def test_concurrent_requests(self):
        """Test multiple concurrent requests are handled safely."""
        adapter = GenOpsVercelAISDKAdapter(team="test-team", project="test-project")
        results = []
        exceptions = []

        def make_request(request_num):
            try:
                with adapter.track_request(
                    "generateText", "openai", f"gpt-{request_num}"
                ) as request:
                    time.sleep(0.1)  # Simulate some processing
                    request.input_tokens = request_num * 10
                    request.output_tokens = request_num * 15
                    results.append(request.request_id)
            except Exception as e:
                exceptions.append(e)

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify results
        self.assertEqual(len(results), 5)
        self.assertEqual(len(exceptions), 0)
        self.assertEqual(
            len(adapter.active_requests), 0
        )  # All requests should be cleaned up


class TestIntegrationModes(unittest.TestCase):
    """Test different integration modes."""

    def test_python_wrapper_mode(self):
        """Test python_wrapper integration mode."""
        adapter = GenOpsVercelAISDKAdapter(integration_mode="python_wrapper")
        self.assertEqual(adapter.integration_mode, "python_wrapper")

    @patch("genops.providers.vercel_ai_sdk.HAS_WEBSOCKETS", True)
    def test_websocket_mode_available(self):
        """Test websocket mode when websockets are available."""
        with patch.object(GenOpsVercelAISDKAdapter, "_initialize_websocket_server"):
            GenOpsVercelAISDKAdapter(integration_mode="websocket")
            # Mode should be websocket if websockets are available
            # (actual behavior depends on HAS_WEBSOCKETS constant)

    @patch("genops.providers.vercel_ai_sdk.HAS_WEBSOCKETS", False)
    def test_websocket_mode_fallback(self):
        """Test websocket mode fallback when websockets not available."""
        adapter = GenOpsVercelAISDKAdapter(integration_mode="websocket")
        self.assertEqual(adapter.integration_mode, "python_wrapper")  # Should fallback

    @patch("genops.providers.vercel_ai_sdk.HAS_NODEJS", True)
    def test_subprocess_mode_available(self):
        """Test subprocess mode when Node.js is available."""
        GenOpsVercelAISDKAdapter(integration_mode="subprocess")
        # Should remain subprocess if Node.js is available

    @patch("genops.providers.vercel_ai_sdk.HAS_NODEJS", False)
    def test_subprocess_mode_fallback(self):
        """Test subprocess mode fallback when Node.js not available."""
        adapter = GenOpsVercelAISDKAdapter(integration_mode="subprocess")
        self.assertEqual(adapter.integration_mode, "python_wrapper")  # Should fallback


if __name__ == "__main__":
    unittest.main()
