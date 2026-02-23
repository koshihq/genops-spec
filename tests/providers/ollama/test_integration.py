"""Integration tests for Ollama provider connectivity and functionality."""

import time
from unittest.mock import Mock, patch

import pytest

from genops.providers.ollama import GenOpsOllamaAdapter, instrument_ollama
from genops.providers.ollama.registration import (
    auto_instrument,
    disable_auto_instrument,
    get_instrumentation_status,
    reset_instrumentation,
)
from genops.providers.ollama.validation import (
    ValidationCategory,
    quick_validate,
    validate_setup,
)


class TestOllamaConnectivity:
    """Test Ollama server connectivity and communication."""

    @pytest.fixture
    def mock_requests_success(self):
        """Mock successful requests."""
        with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
            # Mock version endpoint
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"version": "0.1.17"}

            # Mock generation endpoint
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "response": "Hello! I'm an AI assistant.",
                "eval_count": 10,
                "prompt_eval_count": 5,
            }

            yield mock_get, mock_post

    @pytest.fixture
    def mock_requests_failure(self):
        """Mock failed requests."""
        with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
            import requests

            mock_get.side_effect = requests.exceptions.ConnectionError(
                "Connection refused"
            )
            mock_post.side_effect = requests.exceptions.ConnectionError(
                "Connection refused"
            )

            yield mock_get, mock_post

    def test_successful_connection(self, mock_requests_success):
        """Test successful connection to Ollama server."""
        mock_get, mock_post = mock_requests_success

        # Should not raise exception
        adapter = GenOpsOllamaAdapter(ollama_base_url="http://localhost:11434")
        assert adapter.ollama_base_url == "http://localhost:11434"

    def test_connection_failure(self, mock_requests_failure):
        """Test handling of connection failure."""
        mock_get, mock_post = mock_requests_failure

        with pytest.raises(ConnectionError):
            GenOpsOllamaAdapter(ollama_base_url="http://localhost:11434")

    def test_alternative_url_connection(self, mock_requests_success):
        """Test connection to alternative Ollama URL."""
        mock_get, mock_post = mock_requests_success

        adapter = GenOpsOllamaAdapter(ollama_base_url="http://remote-ollama:11434")
        assert adapter.ollama_base_url == "http://remote-ollama:11434"

    def test_url_normalization(self, mock_requests_success):
        """Test URL normalization (trailing slash removal)."""
        mock_get, mock_post = mock_requests_success

        adapter = GenOpsOllamaAdapter(ollama_base_url="http://localhost:11434/")
        assert adapter.ollama_base_url == "http://localhost:11434"


class TestModelListingIntegration:
    """Test model listing functionality."""

    @pytest.fixture
    def mock_model_response(self):
        """Mock model listing response."""
        return {
            "models": [
                {
                    "name": "llama3.2:1b",
                    "size": 1300000000,
                    "details": {"parameter_size": "1.2B", "family": "llama"},
                },
                {
                    "name": "llama3.2:3b",
                    "size": 3200000000,
                    "details": {"parameter_size": "3.2B", "family": "llama"},
                },
            ]
        }

    def test_list_models_http_api(self, mock_model_response):
        """Test listing models via HTTP API."""
        with patch("requests.get") as mock_get, patch("requests.post"):
            # Mock connection test
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"version": "0.1.17"}

            adapter = GenOpsOllamaAdapter()

            # Mock model listing
            mock_get.return_value.json.return_value = mock_model_response
            models = adapter.list_models()

            assert len(models) == 2
            assert models[0]["name"] == "llama3.2:1b"
            assert models[1]["name"] == "llama3.2:3b"

    def test_list_models_with_governance(self, mock_model_response):
        """Test listing models with governance attributes."""
        with patch("requests.get") as mock_get:
            # Mock connection and model listing
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_model_response

            adapter = GenOpsOllamaAdapter()
            adapter.list_models(team="integration-test", project="model-discovery")

            # Should track the operation
            assert len(adapter.operations) == 1
            operation = adapter.operations[0]
            assert operation.governance_attributes["team"] == "integration-test"
            assert operation.governance_attributes["project"] == "model-discovery"

    def test_list_models_empty_response(self):
        """Test handling of empty model list."""
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"models": []}

            adapter = GenOpsOllamaAdapter()
            models = adapter.list_models()

            assert models == []


class TestGenerationIntegration:
    """Test text generation integration."""

    def test_generate_with_tracking(self):
        """Test text generation with full tracking."""
        with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
            # Mock connection
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"version": "0.1.17"}

            # Mock generation
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "response": "The capital of France is Paris.",
                "eval_count": 8,
                "prompt_eval_count": 6,
            }

            adapter = GenOpsOllamaAdapter(
                cost_tracking_enabled=True, team="integration-test"
            )

            response = adapter.generate(
                model="llama3.2:1b",
                prompt="What is the capital of France?",
                project="qa-testing",
            )

            assert response["response"] == "The capital of France is Paris."

            # Verify operation tracking
            assert len(adapter.operations) == 1
            operation = adapter.operations[0]
            assert operation.operation_type == "generate"
            assert operation.model == "llama3.2:1b"
            assert operation.input_tokens == 6
            assert operation.output_tokens == 8
            assert operation.infrastructure_cost > 0

    def test_chat_with_tracking(self):
        """Test chat functionality with tracking."""
        with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
            # Mock connection
            mock_get.return_value.status_code = 200

            # Mock chat
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "message": {"content": "Hello! How can I help you today?"},
                "eval_count": 9,
            }

            adapter = GenOpsOllamaAdapter()

            messages = [{"role": "user", "content": "Hello"}]
            response = adapter.chat(
                model="llama3.2:3b",
                messages=messages,
                customer_id="integration-customer",
            )

            assert response["message"]["content"] == "Hello! How can I help you today?"

            # Verify tracking
            operation = adapter.operations[0]
            assert operation.operation_type == "chat"
            assert (
                operation.governance_attributes["customer_id"] == "integration-customer"
            )

    def test_generation_error_handling(self):
        """Test error handling during generation."""
        with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
            # Mock connection success
            mock_get.return_value.status_code = 200

            # Mock generation failure
            mock_post.return_value.status_code = 500
            mock_post.return_value.text = "Internal Server Error"
            mock_post.raise_for_status.side_effect = Exception("HTTP 500")

            adapter = GenOpsOllamaAdapter()

            with pytest.raises(Exception):  # noqa: B017
                adapter.generate(model="nonexistent", prompt="test")

            # Should still track failed operation
            assert len(adapter.operations) == 1
            operation = adapter.operations[0]
            assert operation.end_time is not None


class TestValidationIntegration:
    """Test validation system integration."""

    def test_successful_validation(self):
        """Test complete successful validation."""
        with patch("requests.get") as mock_get:
            # Mock all endpoints as successful
            mock_responses = {
                "/api/version": {"version": "0.1.17"},
                "/api/tags": {"models": [{"name": "llama3.2:1b", "size": 1300000000}]},
                "/api/ps": {"models": []},
            }

            def mock_response(*args, **kwargs):
                url = args[0]
                response = Mock()
                response.status_code = 200

                for endpoint, data in mock_responses.items():
                    if endpoint in url:
                        response.json.return_value = data
                        break

                return response

            mock_get.side_effect = mock_response

            result = validate_setup()

            assert result.success
            assert not result.has_critical_issues
            assert result.score > 80  # Should have high score

    def test_validation_with_connection_failure(self):
        """Test validation with connection failure."""
        with patch("requests.get") as mock_get:
            import requests

            mock_get.side_effect = requests.exceptions.ConnectionError()

            result = validate_setup()

            assert not result.success
            assert result.has_critical_issues

            # Should have connection error
            connection_issues = [
                issue
                for issue in result.issues
                if issue.category == ValidationCategory.CONNECTIVITY
            ]
            assert len(connection_issues) > 0

    def test_quick_validate_success(self):
        """Test quick validation success."""
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"version": "0.1.17"}

            result = quick_validate()
            assert result is True

    def test_quick_validate_failure(self):
        """Test quick validation failure."""
        with patch("requests.get") as mock_get:
            import requests

            mock_get.side_effect = requests.exceptions.ConnectionError()

            result = quick_validate()
            assert result is False

    def test_validation_with_missing_dependencies(self):
        """Test validation with missing dependencies."""
        with patch("genops.providers.ollama.validation.HAS_REQUESTS", False):
            result = validate_setup()

            # Should fail due to missing requests
            dependency_issues = [
                issue
                for issue in result.issues
                if issue.category == ValidationCategory.DEPENDENCIES
            ]
            assert len(dependency_issues) > 0


class TestAutoInstrumentationIntegration:
    """Test auto-instrumentation integration."""

    def setUp(self):
        """Reset instrumentation state before each test."""
        reset_instrumentation()

    def tearDown(self):
        """Clean up after each test."""
        reset_instrumentation()

    def test_auto_instrument_without_ollama_client(self):
        """Test auto-instrumentation when Ollama client not available."""
        with patch("genops.providers.ollama.registration.ollama", None):
            result = auto_instrument()
            assert result is False

    @patch("genops.providers.ollama.registration.ollama")
    def test_auto_instrument_with_client(self, mock_ollama):
        """Test auto-instrumentation with Ollama client."""
        # Mock original methods
        mock_ollama.generate = Mock()
        mock_ollama.chat = Mock()

        result = auto_instrument(team="auto-test", project="integration")
        assert result is True

        # Methods should be patched
        assert mock_ollama.generate != Mock()
        assert mock_ollama.chat != Mock()

    def test_instrumentation_status(self):
        """Test getting instrumentation status."""
        status = get_instrumentation_status()

        assert isinstance(status, dict)
        assert "registered" in status
        assert "auto_instrumentation_active" in status
        assert "adapter_configured" in status
        assert "ollama_client_available" in status

    def test_disable_auto_instrument(self):
        """Test disabling auto-instrumentation."""
        with patch("genops.providers.ollama.registration.ollama") as mock_ollama:
            # Enable first
            mock_ollama.generate = Mock()
            original_generate = mock_ollama.generate

            auto_instrument()

            # Methods should be different
            assert mock_ollama.generate != original_generate

            # Disable
            result = disable_auto_instrument()
            assert result is True


class TestInstrumentationFactoryIntegration:
    """Test instrumentation factory functions."""

    def test_instrument_ollama_factory(self):
        """Test instrument_ollama factory function."""
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"version": "0.1.17"}

            adapter = instrument_ollama(
                ollama_base_url="http://test:11434",
                team="factory-test",
                project="integration-test",
                cost_tracking_enabled=True,
            )

            assert isinstance(adapter, GenOpsOllamaAdapter)
            assert adapter.ollama_base_url == "http://test:11434"
            assert adapter.governance_defaults["team"] == "factory-test"
            assert adapter.governance_defaults["project"] == "integration-test"
            assert adapter.cost_tracking_enabled is True


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_complete_workflow(self):
        """Test complete GenOps Ollama workflow."""
        with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
            # Mock connection
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "version": "0.1.17",
                "models": [{"name": "llama3.2:1b", "size": 1300000000}],
            }

            # Mock generation
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "response": "Integration test response",
                "eval_count": 12,
                "prompt_eval_count": 8,
            }

            # 1. Validate setup
            validation_result = validate_setup()
            assert validation_result.success

            # 2. Create adapter
            adapter = instrument_ollama(
                team="integration-team",
                project="end-to-end-test",
                customer_id="test-customer",
                environment="testing",
            )

            # 3. List models
            models = adapter.list_models()
            assert len(models) > 0

            # 4. Generate text
            response = adapter.generate(
                model="llama3.2:1b",
                prompt="This is an integration test",
                priority="high",
            )

            assert "Integration test response" in response["response"]

            # 5. Verify tracking
            operations = adapter.operations
            assert len(operations) == 2  # list_models + generate

            list_op = operations[0]
            assert list_op.operation_type == "list_models"

            gen_op = operations[1]
            assert gen_op.operation_type == "generate"
            assert gen_op.governance_attributes["team"] == "integration-team"
            assert gen_op.governance_attributes["customer_id"] == "test-customer"
            assert gen_op.governance_attributes["priority"] == "high"

            # 6. Get summary
            summary = adapter.get_operation_summary()
            assert summary["total_operations"] == 2
            assert summary["success_rate_percent"] == 100.0
            assert summary["total_infrastructure_cost"] > 0

    def test_error_recovery_workflow(self):
        """Test workflow with errors and recovery."""
        with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
            # Mock connection success
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"version": "0.1.17"}

            adapter = GenOpsOllamaAdapter()

            # First request fails
            mock_post.return_value.status_code = 500
            mock_post.raise_for_status.side_effect = Exception("Server error")

            with pytest.raises(Exception):  # noqa: B017
                adapter.generate(model="test", prompt="fail")

            # Second request succeeds
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"response": "Success"}
            mock_post.raise_for_status.side_effect = None

            response = adapter.generate(model="test", prompt="success")
            assert response["response"] == "Success"

            # Should have tracked both operations
            assert len(adapter.operations) == 2

            summary = adapter.get_operation_summary()
            assert summary["success_rate_percent"] == 50.0  # 1 success, 1 failure


class TestPerformanceIntegration:
    """Test performance aspects of integration."""

    def test_concurrent_operations(self):
        """Test concurrent operation tracking."""
        import concurrent.futures

        with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
            # Mock responses
            mock_get.return_value.status_code = 200
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "response": "Concurrent response"
            }

            adapter = GenOpsOllamaAdapter()

            def generate_text(i):
                return adapter.generate(
                    model=f"model-{i % 3}",
                    prompt=f"Concurrent test {i}",
                    thread_id=str(i),
                )

            # Run 10 concurrent operations
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(generate_text, i) for i in range(10)]
                results = [future.result() for future in futures]

            assert len(results) == 10
            assert len(adapter.operations) == 10

            # All operations should have unique IDs
            operation_ids = [op.operation_id for op in adapter.operations]
            assert len(set(operation_ids)) == 10  # All unique

    def test_large_scale_operation_tracking(self):
        """Test tracking many operations."""
        with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
            mock_get.return_value.status_code = 200
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"response": "Bulk response"}

            adapter = GenOpsOllamaAdapter()

            # Generate many operations
            for i in range(100):
                adapter.generate(
                    model=f"model-{i % 5}",  # 5 different models
                    prompt=f"Bulk test {i}",
                    batch_id=f"batch-{i // 10}",
                )

            assert len(adapter.operations) == 100

            # Test summary performance
            start_time = time.time()
            summary = adapter.get_operation_summary()
            summary_time = time.time() - start_time

            # Summary should be fast even with many operations
            assert summary_time < 1.0  # Should take less than 1 second
            assert summary["total_operations"] == 100
            assert len(summary["models_used"]) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
