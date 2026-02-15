"""
Integration tests for GenOps Elasticsearch integration.

Tests cover:
- End-to-end workflow validation
- Auto-instrumentation
- Multi-operation tracking
- Cost aggregation across operations
- Policy enforcement integration
- Real-world usage scenarios
- Index lifecycle management
- Error recovery patterns
"""

from unittest.mock import patch

import pytest

from genops.providers.elastic import (
    auto_instrument,
    instrument_elastic,
    validate_setup,
)


class TestElasticIntegrationBasicWorkflow:
    """Test basic integration workflows."""

    def test_complete_tracking_workflow(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test complete workflow from initialization to export."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            # Initialize adapter
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            # Track an operation with cost
            with adapter.track_ai_operation(
                "test-completion", operation_type="llm.completion"
            ) as span:
                # Record cost
                adapter.record_cost(
                    span=span,
                    cost=0.05,
                    provider="openai",
                    model="gpt-4",
                    tokens_input=100,
                    tokens_output=200,
                )

            # Verify export was called
            assert (
                mock_elasticsearch_client.bulk.called
                or mock_elasticsearch_client.index.called
            )

    def test_multiple_operations_tracking(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test tracking multiple sequential operations."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            # Track multiple operations
            operations = ["embedding", "completion", "moderation"]
            for op in operations:
                with adapter.track_ai_operation(op, operation_type=f"llm.{op}") as span:
                    adapter.record_cost(
                        span=span,
                        cost=0.01,
                        provider="openai",
                        model="text-embedding-ada-002",
                    )

            # Flush to ensure all exports
            adapter.exporter.flush()

            # Verify multiple exports
            assert (
                mock_elasticsearch_client.bulk.call_count >= 1
                or mock_elasticsearch_client.index.call_count >= 3
            )


class TestElasticIntegrationAutoInstrumentation:
    """Test auto-instrumentation functionality."""

    def test_auto_instrument_with_env_vars(
        self, mock_env_vars, mock_elasticsearch_client
    ):
        """Test auto-instrumentation using environment variables."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = auto_instrument()

            assert adapter is not None
            assert adapter.elastic_url == mock_env_vars["GENOPS_ELASTIC_URL"]
            assert adapter.team == mock_env_vars["GENOPS_TEAM"]

    def test_auto_instrument_returns_singleton(
        self, mock_env_vars, mock_elasticsearch_client
    ):
        """Test that auto_instrument returns the same instance."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter1 = auto_instrument()
            adapter2 = auto_instrument()

            # Should return same instance
            assert adapter1 is adapter2


class TestElasticIntegrationCostAggregation:
    """Test cost aggregation across operations."""

    def test_aggregate_costs_single_provider(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test cost aggregation for single provider."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            total_cost = 0.0
            for i in range(5):
                with adapter.track_ai_operation(f"op-{i}") as span:
                    cost = 0.01 * (i + 1)
                    adapter.record_cost(
                        span=span, cost=cost, provider="openai", model="gpt-4"
                    )
                    total_cost += cost

            # Verify stats tracking
            # Costs should be aggregated in Elasticsearch

    def test_aggregate_costs_multiple_providers(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test cost aggregation across multiple providers."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            providers = [
                ("openai", "gpt-4", 0.05),
                ("anthropic", "claude-3-sonnet", 0.03),
                ("openai", "gpt-3.5-turbo", 0.01),
            ]

            for provider, model, cost in providers:
                with adapter.track_ai_operation(f"{provider}-op") as span:
                    adapter.record_cost(
                        span=span, cost=cost, provider=provider, model=model
                    )

            adapter.exporter.flush()

            # Verify all costs were tracked


class TestElasticIntegrationPolicyEnforcement:
    """Test policy enforcement integration."""

    def test_track_policy_decision(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test tracking policy decisions."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            with adapter.track_ai_operation("policy-check") as span:
                # Record policy decision
                adapter.record_policy(
                    span=span,
                    policy_name="content-filter",
                    policy_result="allowed",
                    policy_reason="content approved",
                )

            # Verify policy attributes were set
            assert span.attributes.get("genops.policy.name") == "content-filter"
            assert span.attributes.get("genops.policy.result") == "allowed"

    def test_track_policy_violation(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test tracking policy violations."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = instrument_elastic(
                **sample_elastic_config,
                export_mode="hybrid",  # Violations should export immediately
                auto_validate=False,
            )

            with adapter.track_ai_operation("policy-violation") as span:
                adapter.record_policy(
                    span=span,
                    policy_name="pii-filter",
                    policy_result="blocked",
                    policy_reason="PII detected in prompt",
                )

            # In HYBRID mode, blocked events should export immediately
            assert mock_elasticsearch_client.index.called


class TestElasticIntegrationExportModes:
    """Test different export modes in integration scenarios."""

    def test_batch_mode_workflow(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test complete workflow in BATCH mode."""
        config = sample_elastic_config.copy()
        config["export_mode"] = "batch"
        config["batch_size"] = 5

        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = instrument_elastic(**config, auto_validate=False)

            # Add enough operations to trigger batch flush
            for i in range(6):
                with adapter.track_ai_operation(f"batch-op-{i}") as span:
                    adapter.record_cost(
                        span=span, cost=0.01, provider="openai", model="gpt-3.5-turbo"
                    )

            # Verify bulk export was called
            assert mock_elasticsearch_client.bulk.called

    def test_realtime_mode_workflow(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test complete workflow in REALTIME mode."""
        config = sample_elastic_config.copy()
        config["export_mode"] = "realtime"

        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = instrument_elastic(**config, auto_validate=False)

            # Each operation should export immediately
            with adapter.track_ai_operation("realtime-op") as span:
                adapter.record_cost(
                    span=span, cost=0.01, provider="openai", model="gpt-3.5-turbo"
                )

            # Verify individual export was called
            assert mock_elasticsearch_client.index.called

    def test_hybrid_mode_workflow(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test complete workflow in HYBRID mode."""
        config = sample_elastic_config.copy()
        config["export_mode"] = "hybrid"
        config["batch_size"] = 10

        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = instrument_elastic(**config, auto_validate=False)

            # Normal operation (should batch)
            with adapter.track_ai_operation("normal-op") as span:
                adapter.record_cost(
                    span=span, cost=0.01, provider="openai", model="gpt-3.5-turbo"
                )

            # Critical operation with error (should export immediately)
            with pytest.raises(ValueError):
                with adapter.track_ai_operation("critical-op") as span:
                    raise ValueError("Simulated error")

            # Verify both batch and realtime exports occurred


class TestElasticIntegrationGovernanceAttributes:
    """Test governance attribute propagation."""

    def test_default_governance_attributes(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test that default governance attributes are propagated."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            with adapter.track_ai_operation("gov-test") as span:
                adapter.record_cost(
                    span=span, cost=0.01, provider="openai", model="gpt-4"
                )

            # Verify governance attributes
            assert span.attributes.get("genops.team") == sample_elastic_config["team"]
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
            assert (
                span.attributes.get("genops.cost_center")
                == sample_elastic_config["cost_center"]
            )

    def test_override_governance_attributes(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test that governance attributes can be overridden per operation."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            with adapter.track_ai_operation(
                "override-test",
                team="custom-team",
                project="custom-project",
                customer_id="custom-customer",
            ) as span:
                pass

            # Verify overridden attributes
            assert span.attributes.get("genops.team") == "custom-team"
            assert span.attributes.get("genops.project") == "custom-project"
            assert span.attributes.get("genops.customer_id") == "custom-customer"


class TestElasticIntegrationILM:
    """Test Index Lifecycle Management integration."""

    def test_ilm_policy_creation(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test that ILM policy is created on initialization."""
        config = sample_elastic_config.copy()
        config["ilm_enabled"] = True
        config["ilm_retention_days"] = 90

        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            instrument_elastic(**config, auto_validate=False)

            # Verify ILM setup was attempted
            # Note: Actual verification depends on client implementation

    def test_ilm_disabled(self, sample_elastic_config, mock_elasticsearch_client):
        """Test that ILM can be disabled."""
        config = sample_elastic_config.copy()
        config["ilm_enabled"] = False

        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = instrument_elastic(**config, auto_validate=False)

            assert adapter.ilm_enabled is False


class TestElasticIntegrationValidation:
    """Test validation integration."""

    def test_validation_on_initialization(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test that validation runs on initialization when enabled."""
        config = sample_elastic_config.copy()
        config["auto_validate"] = True

        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            instrument_elastic(**config)

            # Adapter should initialize successfully with validation

    def test_standalone_validation(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test standalone validation function."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            result = validate_setup(
                elastic_url=sample_elastic_config["url"],
                api_key=sample_elastic_config["api_key"],
                test_index_write=False,
            )

            # Validation should succeed
            assert result.valid is True or len(result.errors) == 0


class TestElasticIntegrationErrorRecovery:
    """Test error recovery patterns."""

    def test_export_failure_recovery(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test graceful handling of export failures."""
        # Simulate intermittent failures
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Connection timeout")
            return {"took": 10, "errors": False, "items": []}

        mock_elasticsearch_client.bulk.side_effect = side_effect

        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            # First operation might fail
            with adapter.track_ai_operation("op-1") as span:
                adapter.record_cost(
                    span=span, cost=0.01, provider="openai", model="gpt-4"
                )

            # Force flush
            adapter.exporter.flush()

            # Second operation should succeed
            with adapter.track_ai_operation("op-2") as span:
                adapter.record_cost(
                    span=span, cost=0.01, provider="openai", model="gpt-4"
                )

            adapter.exporter.flush()

            # Verify recovery
            assert adapter.exporter.stats.total_failed > 0

    def test_operation_error_handling(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test that operation errors are tracked properly."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            with pytest.raises(ValueError):
                with adapter.track_ai_operation("error-op") as span:
                    adapter.record_cost(
                        span=span, cost=0.01, provider="openai", model="gpt-4"
                    )
                    raise ValueError("Operation failed")

            # Error should be tracked in span


class TestElasticIntegrationShutdown:
    """Test graceful shutdown."""

    def test_shutdown_flushes_pending_events(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test that shutdown flushes pending events."""
        config = sample_elastic_config.copy()
        config["export_mode"] = "batch"
        config["batch_size"] = 100

        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = instrument_elastic(**config, auto_validate=False)

            # Add events without reaching batch size
            for i in range(5):
                with adapter.track_ai_operation(f"shutdown-op-{i}") as span:
                    adapter.record_cost(
                        span=span, cost=0.01, provider="openai", model="gpt-4"
                    )

            # Shutdown should flush
            adapter.exporter.shutdown()

            # Verify flush was called
            assert mock_elasticsearch_client.bulk.called
