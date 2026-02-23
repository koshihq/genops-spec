"""
Comprehensive tests for GenOps Elastic adapter implementation.

Tests cover:
- Adapter initialization and configuration
- Context manager lifecycle (track_ai_operation)
- Cost telemetry recording
- Policy enforcement recording
- Export mode configuration
- Error handling and resilience
- Governance attribute propagation
"""

import logging
from unittest.mock import patch

import pytest

from genops.providers.elastic import (
    GenOpsElasticAdapter,
    instrument_elastic,
)
from genops.providers.elastic.event_exporter import ExportMode


class TestElasticAdapterInitialization:
    """Test adapter initialization and configuration."""

    def test_adapter_initialization_with_defaults(
        self, minimal_elastic_config, mock_elasticsearch_client
    ):
        """Test adapter initialization with minimal configuration."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = GenOpsElasticAdapter(
                elastic_url=minimal_elastic_config["url"],
                api_key=minimal_elastic_config["api_key"],
                auto_validate=False,
            )

            assert adapter.elastic_url == "https://localhost:9200"
            assert adapter.api_key == "test-api-key"
            assert adapter.index_prefix == "genops-ai"  # default
            assert adapter.environment == "development"  # default
            assert adapter.export_mode == ExportMode.BATCH  # default

    def test_adapter_initialization_with_full_config(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test adapter initialization with complete configuration."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = GenOpsElasticAdapter(**sample_elastic_config, auto_validate=False)

            assert adapter.elastic_url == sample_elastic_config["url"]
            assert adapter.api_key == sample_elastic_config["api_key"]
            assert adapter.index_prefix == sample_elastic_config["index_prefix"]
            assert adapter.team == sample_elastic_config["team"]
            assert adapter.project == sample_elastic_config["project"]
            assert adapter.environment == sample_elastic_config["environment"]
            assert adapter.customer_id == sample_elastic_config["customer_id"]
            assert adapter.cost_center == sample_elastic_config["cost_center"]
            assert adapter.export_mode == ExportMode.BATCH

    def test_adapter_initialization_with_env_vars(
        self, mock_env_vars, mock_elasticsearch_client
    ):
        """Test adapter initialization with environment variables."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = GenOpsElasticAdapter(auto_validate=False)

            assert adapter.elastic_url == mock_env_vars["GENOPS_ELASTIC_URL"]
            assert adapter.api_key == mock_env_vars["GENOPS_ELASTIC_API_KEY"]
            assert adapter.team == mock_env_vars["GENOPS_TEAM"]
            assert adapter.project == mock_env_vars["GENOPS_PROJECT"]

    def test_adapter_export_mode_validation(
        self, minimal_elastic_config, mock_elasticsearch_client
    ):
        """Test export mode validation and fallback."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            # Valid modes
            for mode in ["batch", "realtime", "hybrid"]:
                adapter = GenOpsElasticAdapter(
                    **minimal_elastic_config, export_mode=mode, auto_validate=False
                )
                assert adapter.export_mode.value == mode

            # Invalid mode should fallback to BATCH with warning
            with patch.object(
                logging.getLogger("genops.providers.elastic.adapter"), "warning"
            ) as mock_warn:
                adapter = GenOpsElasticAdapter(
                    **minimal_elastic_config,
                    export_mode="invalid_mode",
                    auto_validate=False,
                )
                assert adapter.export_mode == ExportMode.BATCH
                mock_warn.assert_called()

    def test_adapter_namespace_fallback(
        self, minimal_elastic_config, mock_elasticsearch_client
    ):
        """Test namespace falls back to team if not specified."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = GenOpsElasticAdapter(
                **minimal_elastic_config, team="test-team", auto_validate=False
            )
            assert adapter.namespace == "test-team"

            adapter_with_namespace = GenOpsElasticAdapter(
                **minimal_elastic_config,
                team="test-team",
                namespace="custom-namespace",
                auto_validate=False,
            )
            assert adapter_with_namespace.namespace == "custom-namespace"


class TestElasticAdapterContextManager:
    """Test track_ai_operation context manager functionality."""

    def test_context_manager_basic_usage(self, mock_elastic_adapter):
        """Test basic context manager usage."""
        with mock_elastic_adapter.track_ai_operation("test-operation") as span:
            assert span is not None
            assert span.name == "test-operation"

    def test_context_manager_with_governance_attributes(self, mock_elastic_adapter):
        """Test context manager with governance attributes."""
        with mock_elastic_adapter.track_ai_operation(
            "test-operation",
            team="custom-team",
            project="custom-project",
            customer_id="custom-customer",
        ) as span:
            # Verify attributes were set
            assert span.attributes.get("genops.team") == "custom-team"
            assert span.attributes.get("genops.project") == "custom-project"
            assert span.attributes.get("genops.customer_id") == "custom-customer"

    def test_context_manager_uses_default_governance_attrs(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test context manager uses adapter default governance attributes."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = GenOpsElasticAdapter(**sample_elastic_config, auto_validate=False)

            with adapter.track_ai_operation("test-operation") as span:
                assert (
                    span.attributes.get("genops.team") == sample_elastic_config["team"]
                )
                assert (
                    span.attributes.get("genops.project")
                    == sample_elastic_config["project"]
                )
                assert (
                    span.attributes.get("genops.environment")
                    == sample_elastic_config["environment"]
                )
                assert (
                    span.attributes.get("genops.customer_id")
                    == sample_elastic_config["customer_id"]
                )

    def test_context_manager_with_custom_attributes(self, mock_elastic_adapter):
        """Test context manager with additional custom attributes."""
        with mock_elastic_adapter.track_ai_operation(
            "test-operation", model_version="v1.0", user_segment="premium"
        ) as span:
            assert span.attributes.get("genops.model_version") == "v1.0"
            assert span.attributes.get("genops.user_segment") == "premium"

    def test_context_manager_error_handling(self, mock_elastic_adapter):
        """Test context manager properly handles exceptions."""
        with pytest.raises(ValueError, match="test error"):
            with mock_elastic_adapter.track_ai_operation("test-operation"):
                raise ValueError("test error")

        # Verify span status was set to ERROR
        # Note: In real implementation, span should be marked with ERROR status

    def test_context_manager_span_export(self, mock_elastic_adapter):
        """Test that span is exported after context manager exits."""
        with patch.object(mock_elastic_adapter.exporter, "export_span") as mock_export:
            with mock_elastic_adapter.track_ai_operation("test-operation"):
                pass

            # Verify export_span was called
            mock_export.assert_called_once()


class TestElasticAdapterCostRecording:
    """Test cost telemetry recording functionality."""

    def test_record_cost_basic(self, mock_elastic_adapter):
        """Test basic cost recording."""
        with mock_elastic_adapter.track_ai_operation("test-operation") as span:
            mock_elastic_adapter.record_cost(
                span=span, cost=0.05, provider="openai", model="gpt-4"
            )

            assert span.attributes.get("genops.cost.total") == 0.05
            assert span.attributes.get("genops.cost.provider") == "openai"
            assert span.attributes.get("genops.cost.model") == "gpt-4"

    def test_record_cost_with_tokens(self, mock_elastic_adapter):
        """Test cost recording with token counts."""
        with mock_elastic_adapter.track_ai_operation("test-operation") as span:
            mock_elastic_adapter.record_cost(
                span=span,
                cost=0.10,
                provider="anthropic",
                model="claude-3-sonnet",
                tokens_input=1000,
                tokens_output=500,
            )

            assert span.attributes.get("genops.cost.total") == 0.10
            assert span.attributes.get("genops.cost.tokens_input") == 1000
            assert span.attributes.get("genops.cost.tokens_output") == 500

    def test_record_cost_with_split_costs(self, mock_elastic_adapter):
        """Test cost recording with separate input/output costs."""
        with mock_elastic_adapter.track_ai_operation("test-operation") as span:
            mock_elastic_adapter.record_cost(
                span=span,
                cost=0.15,
                provider="openai",
                model="gpt-4",
                cost_input=0.10,
                cost_output=0.05,
            )

            assert span.attributes.get("genops.cost.total") == 0.15
            assert span.attributes.get("genops.cost.input") == 0.10
            assert span.attributes.get("genops.cost.output") == 0.05


class TestElasticAdapterPolicyRecording:
    """Test policy enforcement telemetry recording."""

    def test_record_policy_allowed(self, mock_elastic_adapter):
        """Test recording policy decision (allowed)."""
        with mock_elastic_adapter.track_ai_operation("test-operation") as span:
            mock_elastic_adapter.record_policy(
                span=span,
                policy_name="content-filter",
                policy_result="allowed",
                policy_reason="content approved",
            )

            assert span.attributes.get("genops.policy.name") == "content-filter"
            assert span.attributes.get("genops.policy.result") == "allowed"
            assert span.attributes.get("genops.policy.reason") == "content approved"

    def test_record_policy_blocked(self, mock_elastic_adapter):
        """Test recording policy decision (blocked)."""
        with mock_elastic_adapter.track_ai_operation("test-operation") as span:
            mock_elastic_adapter.record_policy(
                span=span,
                policy_name="content-filter",
                policy_result="blocked",
                policy_reason="inappropriate content detected",
            )

            assert span.attributes.get("genops.policy.result") == "blocked"


class TestElasticAdapterExportModes:
    """Test different export modes (BATCH, REALTIME, HYBRID)."""

    def test_batch_mode_initialization(
        self, minimal_elastic_config, mock_elasticsearch_client
    ):
        """Test adapter with BATCH export mode."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = GenOpsElasticAdapter(
                **minimal_elastic_config,
                export_mode="batch",
                batch_size=100,
                batch_interval_seconds=60,
                auto_validate=False,
            )

            assert adapter.export_mode == ExportMode.BATCH
            assert adapter.exporter.batch_size == 100
            assert adapter.exporter.batch_interval_seconds == 60

    def test_realtime_mode_initialization(
        self, minimal_elastic_config, mock_elasticsearch_client
    ):
        """Test adapter with REALTIME export mode."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = GenOpsElasticAdapter(
                **minimal_elastic_config, export_mode="realtime", auto_validate=False
            )

            assert adapter.export_mode == ExportMode.REALTIME

    def test_hybrid_mode_initialization(
        self, minimal_elastic_config, mock_elasticsearch_client
    ):
        """Test adapter with HYBRID export mode."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = GenOpsElasticAdapter(
                **minimal_elastic_config, export_mode="hybrid", auto_validate=False
            )

            assert adapter.export_mode == ExportMode.HYBRID


class TestElasticAdapterInstrumentFunction:
    """Test instrument_elastic factory function."""

    def test_instrument_elastic_basic(
        self, minimal_elastic_config, mock_elasticsearch_client
    ):
        """Test instrument_elastic factory function."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = instrument_elastic(**minimal_elastic_config, auto_validate=False)

            assert isinstance(adapter, GenOpsElasticAdapter)
            assert adapter.elastic_url == minimal_elastic_config["url"]
            assert adapter.api_key == minimal_elastic_config["api_key"]

    def test_instrument_elastic_with_full_config(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test instrument_elastic with complete configuration."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            assert adapter.team == sample_elastic_config["team"]
            assert adapter.project == sample_elastic_config["project"]
            assert adapter.export_mode == ExportMode.BATCH
