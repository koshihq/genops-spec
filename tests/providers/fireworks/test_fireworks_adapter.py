"""
Comprehensive tests for Fireworks AI adapter implementation.

Tests cover:
- Adapter initialization and configuration
- Chat completions with governance
- Embedding operations  
- Session-based tracking
- Error handling and resilience
- Cost attribution and budget management
- Auto-instrumentation functionality
- Fireattention speed optimization validation
"""

import time
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from genops.providers.fireworks import (
    FireworksModel,
    FireworksResult,
    FireworksSessionContext,
    GenOpsFireworksAdapter,
    auto_instrument,
)


class TestFireworksAdapterInitialization:
    """Test adapter initialization and configuration."""

    def test_adapter_initialization_with_defaults(self, sample_fireworks_config):
        """Test adapter initialization with default values."""
        adapter = GenOpsFireworksAdapter(
            team=sample_fireworks_config["team"],
            project=sample_fireworks_config["project"]
        )

        assert adapter.team == "test-team"
        assert adapter.project == "test-project"
        assert adapter.environment == "development"  # default
        assert adapter.daily_budget_limit == 1000.0  # default
        assert adapter.governance_policy == "advisory"  # default
        assert adapter.enable_governance is True

    def test_adapter_initialization_with_custom_config(self, sample_fireworks_config):
        """Test adapter initialization with custom configuration."""
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        assert adapter.team == "test-team"
        assert adapter.project == "test-project"
        assert adapter.environment == "test"
        assert adapter.daily_budget_limit == 100.0
        assert adapter.monthly_budget_limit == 2000.0
        assert adapter.governance_policy == "advisory"
        assert adapter.enable_cost_alerts is True

    def test_adapter_initialization_budget_validation(self):
        """Test budget validation during initialization."""
        with pytest.raises(ValueError, match="Daily budget must be positive"):
            GenOpsFireworksAdapter(
                team="test",
                project="test",
                daily_budget_limit=-100.0
            )

        with pytest.raises(ValueError, match="Monthly budget must be greater than daily"):
            GenOpsFireworksAdapter(
                team="test",
                project="test",
                daily_budget_limit=100.0,
                monthly_budget_limit=50.0
            )

    def test_adapter_initialization_governance_policy_validation(self):
        """Test governance policy validation."""
        valid_policies = ["advisory", "enforcing", "monitoring"]

        for policy in valid_policies:
            adapter = GenOpsFireworksAdapter(
                team="test",
                project="test",
                governance_policy=policy
            )
            assert adapter.governance_policy == policy

        with pytest.raises(ValueError, match="Invalid governance policy"):
            GenOpsFireworksAdapter(
                team="test",
                project="test",
                governance_policy="invalid"
            )

    def test_adapter_client_initialization(self, sample_fireworks_config):
        """Test Fireworks client initialization."""
        with patch('genops.providers.fireworks.Fireworks') as mock_fireworks:
            adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

            # Should initialize client lazily
            assert adapter._client is None

            # Access client property to trigger initialization
            client = adapter.client

            mock_fireworks.assert_called_once()
            assert adapter._client is not None


class TestChatCompletionsWithGovernance:
    """Test chat completion operations with governance tracking."""

    @patch('genops.providers.fireworks.Fireworks')
    def test_chat_with_governance_basic(self, mock_fireworks_class, sample_fireworks_config, sample_chat_messages, mock_fireworks_client):
        """Test basic chat completion with governance."""
        mock_fireworks_class.return_value = mock_fireworks_client

        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        result = adapter.chat_with_governance(
            messages=sample_chat_messages,
            model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
            max_tokens=150,
            feature="test-chat"
        )

        # Verify API call
        mock_fireworks_client.chat.completions.create.assert_called_once()
        call_args = mock_fireworks_client.chat.completions.create.call_args

        assert call_args[1]["model"] == "accounts/fireworks/models/llama-v3p1-8b-instruct"
        assert call_args[1]["messages"] == sample_chat_messages
        assert call_args[1]["max_tokens"] == 150

        # Verify result
        assert isinstance(result, FireworksResult)
        assert result.response == "Test response from Fireworks AI with 4x speed optimization"
        assert result.tokens_used == 75
        assert result.cost > 0
        assert result.execution_time_seconds > 0
        assert result.model_used == "accounts/fireworks/models/llama-v3p1-8b-instruct"

    @patch('genops.providers.fireworks.Fireworks')
    def test_chat_with_governance_attributes(self, mock_fireworks_class, sample_fireworks_config, sample_chat_messages, mock_fireworks_client):
        """Test chat completion with governance attributes."""
        mock_fireworks_class.return_value = mock_fireworks_client

        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        result = adapter.chat_with_governance(
            messages=sample_chat_messages,
            model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
            max_tokens=100,
            feature="test-feature",
            use_case="test-use-case",
            customer_id="test-customer-123",
            cost_center="engineering"
        )

        # Verify governance attributes are captured
        assert result.governance_attrs["feature"] == "test-feature"
        assert result.governance_attrs["use_case"] == "test-use-case"
        assert result.governance_attrs["customer_id"] == "test-customer-123"
        assert result.governance_attrs["cost_center"] == "engineering"
        assert result.governance_attrs["team"] == "test-team"
        assert result.governance_attrs["project"] == "test-project"

    @patch('genops.providers.fireworks.Fireworks')
    def test_chat_with_batch_processing(self, mock_fireworks_class, sample_fireworks_config, sample_chat_messages, mock_fireworks_client):
        """Test chat completion with batch processing discount."""
        mock_fireworks_class.return_value = mock_fireworks_client

        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        result = adapter.chat_with_governance(
            messages=sample_chat_messages,
            model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
            max_tokens=100,
            is_batch=True,
            batch_id="test-batch-123"
        )

        # Verify batch processing is tracked
        assert result.governance_attrs.get("is_batch") is True
        assert result.governance_attrs.get("batch_id") == "test-batch-123"

        # Cost should be reduced by 50% for batch processing
        standard_cost = Decimal("0.015")  # 75 tokens * 0.0002
        expected_batch_cost = standard_cost * Decimal("0.5")
        assert abs(result.cost - expected_batch_cost) < Decimal("0.001")

    @patch('genops.providers.fireworks.Fireworks')
    def test_chat_with_streaming(self, mock_fireworks_class, sample_fireworks_config, sample_chat_messages, mock_fireworks_client):
        """Test chat completion with streaming enabled."""
        # Mock streaming response
        mock_stream = [
            Mock(choices=[Mock(delta=Mock(content="Hello"))]),
            Mock(choices=[Mock(delta=Mock(content=" from"))]),
            Mock(choices=[Mock(delta=Mock(content=" Fireworks"))]),
            Mock(choices=[Mock(delta=Mock(content=" AI!"))])
        ]
        mock_fireworks_client.chat.completions.create.return_value = mock_stream
        mock_fireworks_class.return_value = mock_fireworks_client

        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        chunks_received = []
        def on_chunk(content, cost):
            chunks_received.append((content, cost))

        result = adapter.chat_with_governance(
            messages=sample_chat_messages,
            model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
            max_tokens=100,
            stream=True,
            on_chunk=on_chunk
        )

        # Verify streaming was enabled
        call_args = mock_fireworks_client.chat.completions.create.call_args
        assert call_args[1]["stream"] is True

        # Verify chunk handler was called
        assert len(chunks_received) > 0

    @patch('genops.providers.fireworks.Fireworks')
    def test_chat_error_handling(self, mock_fireworks_class, sample_fireworks_config, sample_chat_messages):
        """Test error handling in chat operations."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_fireworks_class.return_value = mock_client

        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        with pytest.raises(Exception, match="API Error"):
            adapter.chat_with_governance(
                messages=sample_chat_messages,
                model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                max_tokens=100
            )

    @patch('genops.providers.fireworks.Fireworks')
    def test_chat_budget_enforcement(self, mock_fireworks_class, sample_fireworks_config, sample_chat_messages, mock_fireworks_client):
        """Test budget enforcement in enforcing governance policy."""
        config = sample_fireworks_config.copy()
        config["governance_policy"] = "enforcing"
        config["daily_budget_limit"] = 0.001  # Very low budget

        mock_fireworks_class.return_value = mock_fireworks_client
        adapter = GenOpsFireworksAdapter(**config)

        # Mock the adapter to track high spending
        adapter._daily_costs = Decimal("0.001")  # Already at budget limit

        with pytest.raises(Exception, match="Budget exceeded"):
            adapter.chat_with_governance(
                messages=sample_chat_messages,
                model=FireworksModel.LLAMA_3_1_70B_INSTRUCT,  # Expensive model
                max_tokens=500  # High token count
            )


class TestEmbeddingOperations:
    """Test embedding operations with governance."""

    @patch('genops.providers.fireworks.Fireworks')
    def test_embeddings_with_governance_basic(self, mock_fireworks_class, sample_fireworks_config, sample_embedding_texts, mock_fireworks_client):
        """Test basic embedding generation with governance."""
        mock_fireworks_class.return_value = mock_fireworks_client

        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        result = adapter.embeddings_with_governance(
            input_texts=sample_embedding_texts,
            model=FireworksModel.NOMIC_EMBED_TEXT,
            feature="test-embeddings"
        )

        # Verify API call
        mock_fireworks_client.embeddings.create.assert_called_once()
        call_args = mock_fireworks_client.embeddings.create.call_args

        assert call_args[1]["model"] == "accounts/fireworks/models/nomic-embed-text-v1p5"
        assert call_args[1]["input"] == sample_embedding_texts

        # Verify result
        assert isinstance(result, FireworksResult)
        assert result.embeddings is not None
        assert len(result.embeddings) == 2  # Mock returns 2 embeddings
        assert result.tokens_used == 100
        assert result.cost > 0

    @patch('genops.providers.fireworks.Fireworks')
    def test_embeddings_error_handling(self, mock_fireworks_class, sample_fireworks_config, sample_embedding_texts):
        """Test error handling in embedding operations."""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("Embedding API Error")
        mock_fireworks_class.return_value = mock_client

        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        with pytest.raises(Exception, match="Embedding API Error"):
            adapter.embeddings_with_governance(
                input_texts=sample_embedding_texts,
                model=FireworksModel.NOMIC_EMBED_TEXT
            )


class TestSessionBasedTracking:
    """Test session-based operation tracking."""

    def test_track_session_context_manager(self, sample_fireworks_config):
        """Test session context manager lifecycle."""
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        with adapter.track_session("test-session") as session:
            assert isinstance(session, FireworksSessionContext)
            assert session.session_name == "test-session"
            assert session.session_id is not None
            assert session.start_time > 0
            assert session.total_operations == 0
            assert session.total_cost == Decimal("0")

        # Session should be finalized after context exit
        assert session.end_time is not None

    def test_track_session_with_governance_attrs(self, sample_fireworks_config):
        """Test session tracking with governance attributes."""
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        with adapter.track_session(
            "test-session",
            customer_id="test-customer",
            use_case="testing",
            cost_center="engineering"
        ) as session:
            assert session.governance_attrs["customer_id"] == "test-customer"
            assert session.governance_attrs["use_case"] == "testing"
            assert session.governance_attrs["cost_center"] == "engineering"

    @patch('genops.providers.fireworks.Fireworks')
    def test_session_operation_tracking(self, mock_fireworks_class, sample_fireworks_config, sample_chat_messages, mock_fireworks_client):
        """Test operation tracking within a session."""
        mock_fireworks_class.return_value = mock_fireworks_client

        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        with adapter.track_session("test-session") as session:
            result = adapter.chat_with_governance(
                messages=sample_chat_messages,
                model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                max_tokens=100,
                session_id=session.session_id
            )

            assert session.total_operations == 1
            assert session.total_cost > 0
            assert result.governance_attrs["session_id"] == session.session_id


class TestAutoInstrumentation:
    """Test auto-instrumentation functionality."""

    @patch('genops.providers.fireworks.Fireworks')
    def test_auto_instrument_function(self, mock_fireworks_class):
        """Test auto-instrumentation activation."""
        # Mock the auto-instrumentation setup
        with patch('genops.providers.fireworks._setup_auto_instrumentation') as mock_setup:
            auto_instrument()

            mock_setup.assert_called_once()

    def test_auto_instrument_with_config(self):
        """Test auto-instrumentation with custom configuration."""
        config = {
            "team": "auto-team",
            "project": "auto-project",
            "daily_budget_limit": 50.0
        }

        with patch('genops.providers.fireworks._setup_auto_instrumentation') as mock_setup:
            auto_instrument(**config)

            mock_setup.assert_called_once_with(**config)


class TestCostManagement:
    """Test cost calculation and budget management."""

    def test_get_cost_summary(self, sample_fireworks_config, mock_cost_summary):
        """Test cost summary retrieval."""
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        # Mock internal cost tracking
        adapter._daily_costs = mock_cost_summary["daily_costs"]
        adapter._monthly_costs = mock_cost_summary["monthly_costs"]
        adapter._operations_count = mock_cost_summary["operations_count"]

        summary = adapter.get_cost_summary()

        assert summary["daily_costs"] == mock_cost_summary["daily_costs"]
        assert summary["daily_budget_utilization"] == 5.25  # 5.25/100.0 * 100
        assert summary["operations_count"] == 150

    def test_cost_calculation_accuracy(self, sample_fireworks_config):
        """Test cost calculation accuracy across models."""
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        # Test different models and token counts
        test_cases = [
            (FireworksModel.LLAMA_3_2_1B_INSTRUCT, 1000, Decimal("0.0001")),  # 1000 * 0.0001/1000
            (FireworksModel.LLAMA_3_1_8B_INSTRUCT, 1000, Decimal("0.0002")),  # 1000 * 0.0002/1000
            (FireworksModel.LLAMA_3_1_70B_INSTRUCT, 1000, Decimal("0.0009"))  # 1000 * 0.0009/1000
        ]

        for model, tokens, expected_cost in test_cases:
            calculated_cost = adapter._calculate_cost(model.value, tokens, is_batch=False)
            assert abs(calculated_cost - expected_cost) < Decimal("0.0001")

    def test_batch_cost_discount(self, sample_fireworks_config):
        """Test batch processing cost discount."""
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        model = FireworksModel.LLAMA_3_1_8B_INSTRUCT.value
        tokens = 1000

        standard_cost = adapter._calculate_cost(model, tokens, is_batch=False)
        batch_cost = adapter._calculate_cost(model, tokens, is_batch=True)

        # Batch should be 50% of standard cost
        expected_batch_cost = standard_cost * Decimal("0.5")
        assert abs(batch_cost - expected_batch_cost) < Decimal("0.0001")


class TestPerformanceAndOptimization:
    """Test performance features and Fireattention optimization."""

    @patch('genops.providers.fireworks.Fireworks')
    def test_fireattention_speed_tracking(self, mock_fireworks_class, sample_fireworks_config, sample_chat_messages, mock_fireworks_client):
        """Test Fireattention speed optimization tracking."""
        mock_fireworks_class.return_value = mock_fireworks_client

        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        start_time = time.time()
        result = adapter.chat_with_governance(
            messages=sample_chat_messages,
            model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
            max_tokens=100
        )

        # Verify speed optimization is tracked
        assert result.execution_time_seconds > 0
        assert result.execution_time_seconds < 5.0  # Should be fast with Fireattention

        # Verify Fireattention optimization is flagged
        assert result.governance_attrs.get("fireattention_optimized") is True

    def test_performance_metrics_collection(self, sample_fireworks_config):
        """Test collection of performance metrics."""
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        # Mock performance data
        adapter._performance_metrics = {
            "avg_response_time": 0.85,
            "tokens_per_second": 120,
            "fireattention_speedup": 4.0
        }

        metrics = adapter.get_performance_metrics()

        assert metrics["avg_response_time"] == 0.85
        assert metrics["tokens_per_second"] == 120
        assert metrics["fireattention_speedup"] == 4.0


class TestErrorHandlingAndResilience:
    """Test error handling and resilience patterns."""

    @patch('genops.providers.fireworks.Fireworks')
    def test_api_timeout_handling(self, mock_fireworks_class, sample_fireworks_config, sample_chat_messages):
        """Test API timeout handling."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = TimeoutError("Request timeout")
        mock_fireworks_class.return_value = mock_client

        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        with pytest.raises(TimeoutError, match="Request timeout"):
            adapter.chat_with_governance(
                messages=sample_chat_messages,
                model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                max_tokens=100
            )

    @patch('genops.providers.fireworks.Fireworks')
    def test_rate_limit_handling(self, mock_fireworks_class, sample_fireworks_config, sample_chat_messages):
        """Test rate limit handling."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("Rate limit exceeded")
        mock_fireworks_class.return_value = mock_client

        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        with pytest.raises(Exception, match="Rate limit exceeded"):
            adapter.chat_with_governance(
                messages=sample_chat_messages,
                model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                max_tokens=100
            )

    def test_invalid_model_handling(self, sample_fireworks_config, sample_chat_messages):
        """Test handling of invalid model specifications."""
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        with pytest.raises(ValueError, match="Invalid model"):
            adapter.chat_with_governance(
                messages=sample_chat_messages,
                model="invalid-model-name",
                max_tokens=100
            )


class TestFireworksModels:
    """Test Fireworks model enumeration and validation."""

    def test_model_enum_values(self):
        """Test that all expected models are available."""
        expected_models = [
            "LLAMA_3_2_1B_INSTRUCT",
            "LLAMA_3_1_8B_INSTRUCT",
            "LLAMA_3_1_70B_INSTRUCT",
            "LLAMA_3_1_405B_INSTRUCT",
            "MIXTRAL_8X7B",
            "DEEPSEEK_CODER_V2_LITE",
            "DEEPSEEK_R1_DISTILL",
            "NOMIC_EMBED_TEXT",
            "LLAMA_VISION_11B"
        ]

        for model_name in expected_models:
            assert hasattr(FireworksModel, model_name)
            model = getattr(FireworksModel, model_name)
            assert model.value.startswith("accounts/fireworks/models/")

    def test_model_pricing_tiers(self):
        """Test model pricing tier classification."""
        # Test various pricing tiers
        tiny_models = [FireworksModel.LLAMA_3_2_1B_INSTRUCT]
        small_models = [FireworksModel.LLAMA_3_1_8B_INSTRUCT]
        large_models = [FireworksModel.LLAMA_3_1_70B_INSTRUCT, FireworksModel.LLAMA_3_1_405B_INSTRUCT]

        # All models should have valid enum values
        for model_list in [tiny_models, small_models, large_models]:
            for model in model_list:
                assert isinstance(model.value, str)
                assert len(model.value) > 0
