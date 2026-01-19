"""Unit tests for Collibra asset exporter."""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from genops.providers.collibra.asset_exporter import AssetExporter, ExportMode
from genops.providers.collibra.client import CollibraAPIClient
from tests.mocks.mock_collibra_server import MockCollibraServer


@pytest.fixture
def mock_client():
    """Create mock Collibra client."""
    client = MagicMock(spec=CollibraAPIClient)
    client.create_asset = Mock(return_value={"id": "asset-123", "name": "Test Asset"})
    return client


@pytest.fixture
def mock_server():
    """Create mock Collibra server."""
    server = MockCollibraServer()
    yield server
    server.reset()


def test_exporter_initialization_batch_mode(mock_client):
    """Test exporter initialization with batch mode."""
    exporter = AssetExporter(
        client=mock_client,
        domain_id="domain-123",
        export_mode=ExportMode.BATCH,
        batch_size=50,
        batch_interval_seconds=30,
    )

    assert exporter.export_mode == ExportMode.BATCH
    assert exporter.batch_size == 50
    assert exporter.batch_interval_seconds == 30
    assert exporter.domain_id == "domain-123"


def test_exporter_initialization_realtime_mode(mock_client):
    """Test exporter initialization with real-time mode."""
    exporter = AssetExporter(
        client=mock_client,
        domain_id="domain-123",
        export_mode=ExportMode.REALTIME,
        enable_background_flush=False,
    )

    assert exporter.export_mode == ExportMode.REALTIME
    assert exporter.background_thread is None


def test_export_span_realtime_mode(mock_client):
    """Test exporting span in real-time mode."""
    exporter = AssetExporter(
        client=mock_client,
        domain_id="domain-123",
        export_mode=ExportMode.REALTIME,
        enable_background_flush=False,
    )

    span_attributes = {
        "genops.cost.total": 0.05,
        "genops.cost.provider": "openai",
        "genops.operation.name": "completion",
    }

    result = exporter.export_span(span_attributes)

    # Should call client immediately
    assert mock_client.create_asset.called
    assert result is not None
    assert result["id"] == "asset-123"

    # Check statistics
    stats = exporter.get_stats()
    assert stats.assets_exported == 1
    assert stats.assets_failed == 0


def test_export_span_batch_mode(mock_client):
    """Test exporting span in batch mode."""
    exporter = AssetExporter(
        client=mock_client,
        domain_id="domain-123",
        export_mode=ExportMode.BATCH,
        batch_size=10,
        enable_background_flush=False,
    )

    span_attributes = {
        "genops.cost.total": 0.05,
        "genops.operation.name": "completion",
    }

    result = exporter.export_span(span_attributes)

    # Should not call client yet (buffered)
    assert not mock_client.create_asset.called
    assert result is None

    # Check buffer
    assert exporter.get_buffer_size() == 1


def test_batch_auto_flush_when_size_reached(mock_client):
    """Test automatic batch flush when size limit reached."""
    exporter = AssetExporter(
        client=mock_client,
        domain_id="domain-123",
        export_mode=ExportMode.BATCH,
        batch_size=3,
        enable_background_flush=False,
    )

    # Add 3 spans (should trigger flush)
    for i in range(3):
        exporter.export_span(
            {"genops.operation.name": f"operation-{i}", "genops.cost.total": 0.01}
        )

    # Should have flushed
    assert mock_client.create_asset.call_count == 3
    assert exporter.get_buffer_size() == 0


def test_manual_flush(mock_client):
    """Test manual flush of batch buffer."""
    exporter = AssetExporter(
        client=mock_client,
        domain_id="domain-123",
        export_mode=ExportMode.BATCH,
        batch_size=100,
        enable_background_flush=False,
    )

    # Add 5 spans
    for i in range(5):
        exporter.export_span({"genops.operation.name": f"operation-{i}"})

    assert exporter.get_buffer_size() == 5

    # Manual flush
    count = exporter.flush()

    assert count == 5
    assert exporter.get_buffer_size() == 0
    assert mock_client.create_asset.call_count == 5


def test_hybrid_mode_critical_event_realtime(mock_client):
    """Test hybrid mode exports critical events in real-time."""
    exporter = AssetExporter(
        client=mock_client,
        domain_id="domain-123",
        export_mode=ExportMode.HYBRID,
        enable_background_flush=False,
    )

    # Critical event: policy blocked
    span_attributes = {
        "genops.operation.name": "blocked-operation",
        "genops.policy.result": "blocked",
    }

    result = exporter.export_span(span_attributes)

    # Should export immediately for critical event
    assert mock_client.create_asset.called
    assert result is not None


def test_hybrid_mode_regular_event_batched(mock_client):
    """Test hybrid mode batches regular events."""
    exporter = AssetExporter(
        client=mock_client,
        domain_id="domain-123",
        export_mode=ExportMode.HYBRID,
        enable_background_flush=False,
    )

    # Regular event
    span_attributes = {
        "genops.operation.name": "regular-operation",
        "genops.cost.total": 0.01,
    }

    result = exporter.export_span(span_attributes)

    # Should not export immediately
    assert not mock_client.create_asset.called
    assert result is None
    assert exporter.get_buffer_size() == 1


def test_hybrid_mode_high_cost_realtime(mock_client):
    """Test hybrid mode exports high-cost operations in real-time."""
    exporter = AssetExporter(
        client=mock_client,
        domain_id="domain-123",
        export_mode=ExportMode.HYBRID,
        enable_background_flush=False,
    )

    # High-cost operation (>$10)
    span_attributes = {
        "genops.operation.name": "expensive-operation",
        "genops.cost.total": 15.0,
    }

    result = exporter.export_span(span_attributes)

    # Should export immediately
    assert mock_client.create_asset.called
    assert result is not None


def test_export_statistics_tracking(mock_client):
    """Test export statistics are tracked correctly."""
    exporter = AssetExporter(
        client=mock_client,
        domain_id="domain-123",
        export_mode=ExportMode.REALTIME,
        enable_background_flush=False,
    )

    # Export 3 successful operations
    for i in range(3):
        exporter.export_span({"genops.operation.name": f"op-{i}"})

    stats = exporter.get_stats()

    assert stats.assets_exported == 3
    assert stats.assets_failed == 0
    assert stats.total_export_time_ms > 0
    assert stats.last_export_time is not None


def test_export_failure_handling(mock_client):
    """Test export failure is handled gracefully."""
    # Configure client to raise error
    mock_client.create_asset.side_effect = Exception("API Error")

    exporter = AssetExporter(
        client=mock_client,
        domain_id="domain-123",
        export_mode=ExportMode.REALTIME,
        enable_background_flush=False,
    )

    # Export should not raise exception
    result = exporter.export_span({"genops.operation.name": "test"})

    assert result is None

    stats = exporter.get_stats()
    assert stats.assets_exported == 0
    assert stats.assets_failed == 1


def test_shutdown_flushes_remaining_data(mock_client):
    """Test shutdown flushes remaining buffered data."""
    exporter = AssetExporter(
        client=mock_client,
        domain_id="domain-123",
        export_mode=ExportMode.BATCH,
        batch_size=100,
        enable_background_flush=False,
    )

    # Add some spans
    for i in range(5):
        exporter.export_span({"genops.operation.name": f"op-{i}"})

    assert exporter.get_buffer_size() == 5

    # Shutdown
    exporter.shutdown()

    # Should have flushed remaining data
    assert mock_client.create_asset.call_count == 5
    assert exporter.get_buffer_size() == 0


def test_background_flush_not_started_in_realtime_mode(mock_client):
    """Test background thread not started in real-time mode."""
    exporter = AssetExporter(
        client=mock_client,
        domain_id="domain-123",
        export_mode=ExportMode.REALTIME,
        enable_background_flush=True,  # Requested but shouldn't start
    )

    # Background thread should not start for real-time mode
    assert exporter.background_thread is None or not exporter.background_thread.is_alive()


def test_is_critical_event_policy_blocked():
    """Test critical event detection for blocked policies."""
    exporter = AssetExporter(
        client=MagicMock(),
        domain_id="domain-123",
        export_mode=ExportMode.HYBRID,
        enable_background_flush=False,
    )

    assert exporter._is_critical_event({"genops.policy.result": "blocked"})
    assert exporter._is_critical_event({"genops.policy.result": "rate_limited"})
    assert not exporter._is_critical_event({"genops.policy.result": "allowed"})


def test_is_critical_event_high_cost():
    """Test critical event detection for high-cost operations."""
    exporter = AssetExporter(
        client=MagicMock(),
        domain_id="domain-123",
        export_mode=ExportMode.HYBRID,
        enable_background_flush=False,
    )

    assert exporter._is_critical_event({"genops.cost.total": 15.0})
    assert exporter._is_critical_event({"genops.cost.total": 10.1})
    assert not exporter._is_critical_event({"genops.cost.total": 5.0})


def test_is_critical_event_budget_exceeded():
    """Test critical event detection for budget exceeded."""
    exporter = AssetExporter(
        client=MagicMock(),
        domain_id="domain-123",
        export_mode=ExportMode.HYBRID,
        enable_background_flush=False,
    )

    assert exporter._is_critical_event({"genops.budget.remaining": 0})
    assert exporter._is_critical_event({"genops.budget.remaining": -10.0})
    assert not exporter._is_critical_event({"genops.budget.remaining": 50.0})
