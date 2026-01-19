"""
Comprehensive tests for Elastic event exporter.

Tests cover:
- Export mode configuration (BATCH, REALTIME, HYBRID)
- Batch buffering and flushing
- Background thread management
- Export statistics tracking
- Error handling and resilience
- Thread safety
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time
import threading

from genops.providers.elastic.event_exporter import (
    EventExporter,
    ExportMode,
    ExportStats,
)
from genops.providers.elastic.client import ElasticAPIClient


class TestEventExporterInitialization:
    """Test event exporter initialization."""

    def test_exporter_initialization_batch_mode(self, mock_elasticsearch_client):
        """Test exporter initialization in BATCH mode."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key"
            )

            exporter = EventExporter(
                client=client,
                export_mode=ExportMode.BATCH,
                batch_size=100,
                batch_interval_seconds=60,
                enable_background_flush=False  # Disable for testing
            )

            assert exporter.export_mode == ExportMode.BATCH
            assert exporter.batch_size == 100
            assert exporter.batch_interval_seconds == 60

    def test_exporter_initialization_realtime_mode(self, mock_elasticsearch_client):
        """Test exporter initialization in REALTIME mode."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key"
            )

            exporter = EventExporter(
                client=client,
                export_mode=ExportMode.REALTIME,
                enable_background_flush=False
            )

            assert exporter.export_mode == ExportMode.REALTIME

    def test_exporter_initialization_hybrid_mode(self, mock_elasticsearch_client):
        """Test exporter initialization in HYBRID mode."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key"
            )

            exporter = EventExporter(
                client=client,
                export_mode=ExportMode.HYBRID,
                enable_background_flush=False
            )

            assert exporter.export_mode == ExportMode.HYBRID


class TestEventExporterBatchMode:
    """Test batch mode export functionality."""

    def test_batch_mode_buffers_events(self, mock_elasticsearch_client, sample_telemetry_event):
        """Test that batch mode buffers events before flushing."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key"
            )

            exporter = EventExporter(
                client=client,
                export_mode=ExportMode.BATCH,
                batch_size=10,
                enable_background_flush=False
            )

            # Add events without reaching batch_size
            for i in range(5):
                exporter.export_span(sample_telemetry_event, is_critical=False)

            # Should be buffered, not exported yet
            assert len(exporter.event_buffer) == 5
            mock_elasticsearch_client.bulk.assert_not_called()

    def test_batch_mode_flushes_when_full(self, mock_elasticsearch_client, sample_telemetry_event):
        """Test that batch mode flushes when batch_size is reached."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key"
            )

            exporter = EventExporter(
                client=client,
                export_mode=ExportMode.BATCH,
                batch_size=5,
                enable_background_flush=False
            )

            # Add exactly batch_size events
            for i in range(5):
                exporter.export_span(sample_telemetry_event, is_critical=False)

            # Should have flushed
            mock_elasticsearch_client.bulk.assert_called()

    def test_batch_mode_manual_flush(self, mock_elasticsearch_client, sample_telemetry_event):
        """Test manual flush in batch mode."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key"
            )

            exporter = EventExporter(
                client=client,
                export_mode=ExportMode.BATCH,
                batch_size=100,
                enable_background_flush=False
            )

            # Add a few events
            for i in range(3):
                exporter.export_span(sample_telemetry_event, is_critical=False)

            # Manually flush
            exporter.flush()

            # Should have exported
            mock_elasticsearch_client.bulk.assert_called_once()
            assert len(exporter.event_buffer) == 0


class TestEventExporterRealtimeMode:
    """Test realtime mode export functionality."""

    def test_realtime_mode_exports_immediately(self, mock_elasticsearch_client, sample_telemetry_event):
        """Test that realtime mode exports each event immediately."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key"
            )

            exporter = EventExporter(
                client=client,
                export_mode=ExportMode.REALTIME,
                enable_background_flush=False
            )

            # Export single event
            exporter.export_span(sample_telemetry_event, is_critical=False)

            # Should export immediately
            mock_elasticsearch_client.index.assert_called_once()

    def test_realtime_mode_multiple_events(self, mock_elasticsearch_client, sample_batch_events):
        """Test realtime mode with multiple events."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key"
            )

            exporter = EventExporter(
                client=client,
                export_mode=ExportMode.REALTIME,
                enable_background_flush=False
            )

            # Export multiple events
            for event in sample_batch_events:
                exporter.export_span(event, is_critical=False)

            # Each should be exported immediately
            assert mock_elasticsearch_client.index.call_count == len(sample_batch_events)


class TestEventExporterHybridMode:
    """Test hybrid mode export functionality."""

    def test_hybrid_mode_critical_events_immediate(self, mock_elasticsearch_client, sample_telemetry_event):
        """Test that hybrid mode exports critical events immediately."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key"
            )

            exporter = EventExporter(
                client=client,
                export_mode=ExportMode.HYBRID,
                batch_size=10,
                enable_background_flush=False
            )

            # Export critical event
            exporter.export_span(sample_telemetry_event, is_critical=True)

            # Should export immediately
            mock_elasticsearch_client.index.assert_called_once()

    def test_hybrid_mode_normal_events_batched(self, mock_elasticsearch_client, sample_telemetry_event):
        """Test that hybrid mode batches normal events."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key"
            )

            exporter = EventExporter(
                client=client,
                export_mode=ExportMode.HYBRID,
                batch_size=10,
                enable_background_flush=False
            )

            # Export normal events
            for i in range(3):
                exporter.export_span(sample_telemetry_event, is_critical=False)

            # Should be buffered
            assert len(exporter.event_buffer) == 3
            mock_elasticsearch_client.bulk.assert_not_called()


class TestEventExporterStatistics:
    """Test export statistics tracking."""

    def test_export_stats_initialization(self):
        """Test export statistics initialization."""
        stats = ExportStats()

        assert stats.total_exported == 0
        assert stats.total_failed == 0
        assert stats.total_batches == 0
        assert stats.total_realtime == 0

    def test_export_stats_record_success(self):
        """Test recording successful exports."""
        stats = ExportStats()

        stats.record_success(count=10, duration_ms=100.0, is_batch=True)

        assert stats.total_exported == 10
        assert stats.total_batches == 1
        assert stats.last_batch_size == 10
        assert stats.last_export_duration_ms == 100.0

    def test_export_stats_record_failure(self):
        """Test recording failed exports."""
        stats = ExportStats()

        stats.record_failure("Connection timeout")

        assert stats.total_failed == 1
        assert len(stats.errors) == 1
        assert "Connection timeout" in stats.errors[0]

    def test_export_stats_to_dict(self):
        """Test converting stats to dictionary."""
        stats = ExportStats()
        stats.record_success(count=5, duration_ms=50.0, is_batch=True)

        stats_dict = stats.to_dict()

        assert stats_dict["total_exported"] == 5
        assert stats_dict["total_batches"] == 1
        assert stats_dict["last_export_duration_ms"] == 50.0


class TestEventExporterBackgroundFlush:
    """Test background flush thread functionality."""

    def test_background_flush_thread_starts(self, mock_elasticsearch_client):
        """Test that background flush thread starts in batch mode."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key"
            )

            exporter = EventExporter(
                client=client,
                export_mode=ExportMode.BATCH,
                batch_interval_seconds=1,
                enable_background_flush=True
            )

            # Verify flush thread is running
            assert exporter.flush_thread is not None
            assert exporter.flush_thread.is_alive()

            # Cleanup
            exporter.shutdown()

    def test_background_flush_shutdown(self, mock_elasticsearch_client):
        """Test graceful shutdown of background flush thread."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key"
            )

            exporter = EventExporter(
                client=client,
                export_mode=ExportMode.BATCH,
                enable_background_flush=True
            )

            # Shutdown and verify thread stops
            exporter.shutdown()
            time.sleep(0.2)  # Give thread time to stop

            assert not exporter.flush_thread.is_alive()


class TestEventExporterErrorHandling:
    """Test error handling and resilience."""

    def test_export_handles_connection_errors(self, mock_elasticsearch_client, sample_telemetry_event):
        """Test that export handles connection errors gracefully."""
        mock_elasticsearch_client.index.side_effect = Exception("Connection lost")

        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key"
            )

            exporter = EventExporter(
                client=client,
                export_mode=ExportMode.REALTIME,
                enable_background_flush=False
            )

            # Should not raise, but record error
            exporter.export_span(sample_telemetry_event, is_critical=False)

            # Verify error was recorded
            assert exporter.stats.total_failed > 0
