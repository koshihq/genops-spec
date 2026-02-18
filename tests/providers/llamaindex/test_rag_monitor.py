"""
Unit tests for GenOps LlamaIndex RAG Monitor.

Comprehensive test coverage for RAG pipeline monitoring including
quality metrics, performance tracking, and operation monitoring.
"""

import time

import pytest

# Import the module under test
from genops.providers.llamaindex.rag_monitor import (
    RAGMonitor,
    RAGOperationAnalytics,
    RAGPerformanceMetrics,
    RAGQualityMetrics,
    create_rag_monitor,
)


class TestRAGMonitorInitialization:
    """Test RAG monitor initialization and configuration."""

    def test_default_initialization(self):
        """Test RAG monitor with default parameters."""
        monitor = RAGMonitor()

        assert monitor.enable_quality_metrics is True
        assert monitor.enable_cost_tracking is True
        assert monitor.enable_performance_profiling is True
        assert monitor.operations == []
        assert monitor.quality_scores == []
        assert monitor.performance_metrics == []

    def test_initialization_with_disabled_features(self):
        """Test RAG monitor with disabled features."""
        monitor = RAGMonitor(
            enable_quality_metrics=False,
            enable_cost_tracking=False,
            enable_performance_profiling=False,
        )

        assert monitor.enable_quality_metrics is False
        assert monitor.enable_cost_tracking is False
        assert monitor.enable_performance_profiling is False

    def test_initialization_with_governance_attributes(self):
        """Test RAG monitor with governance attributes."""
        governance_attrs = {
            "team": "rag-team",
            "project": "rag-project",
            "environment": "test",
        }

        monitor = RAGMonitor(**governance_attrs)
        assert monitor.default_governance_attrs == governance_attrs


class TestRAGQualityMetrics:
    """Test RAG quality metrics dataclass."""

    def test_quality_metrics_initialization(self):
        """Test quality metrics initialization with default values."""
        metrics = RAGQualityMetrics()

        assert metrics.retrieval_relevance == 0.0
        assert metrics.response_faithfulness == 0.0
        assert metrics.answer_relevancy == 0.0
        assert metrics.context_precision == 0.0
        assert metrics.context_recall == 0.0
        assert metrics.semantic_similarity == 0.0
        assert metrics.factual_consistency == 0.0

    def test_quality_metrics_with_values(self):
        """Test quality metrics with specific values."""
        metrics = RAGQualityMetrics(
            retrieval_relevance=0.85,
            response_faithfulness=0.90,
            answer_relevancy=0.88,
            context_precision=0.82,
            context_recall=0.87,
            semantic_similarity=0.91,
            factual_consistency=0.89,
        )

        assert metrics.retrieval_relevance == 0.85
        assert metrics.response_faithfulness == 0.90
        assert metrics.answer_relevancy == 0.88
        assert metrics.context_precision == 0.82
        assert metrics.context_recall == 0.87
        assert metrics.semantic_similarity == 0.91
        assert metrics.factual_consistency == 0.89

    def test_quality_metrics_average_score(self):
        """Test quality metrics average score calculation."""
        metrics = RAGQualityMetrics(
            retrieval_relevance=0.8,
            response_faithfulness=0.9,
            answer_relevancy=0.85,
            context_precision=0.75,
            context_recall=0.8,
            semantic_similarity=0.85,
            factual_consistency=0.9,
        )

        expected_avg = (0.8 + 0.9 + 0.85 + 0.75 + 0.8 + 0.85 + 0.9) / 7
        assert abs(metrics.average_score() - expected_avg) < 1e-10


class TestRAGPerformanceMetrics:
    """Test RAG performance metrics dataclass."""

    def test_performance_metrics_initialization(self):
        """Test performance metrics initialization with default values."""
        metrics = RAGPerformanceMetrics()

        assert metrics.embedding_latency_ms == 0.0
        assert metrics.retrieval_latency_ms == 0.0
        assert metrics.synthesis_latency_ms == 0.0
        assert metrics.total_latency_ms == 0.0
        assert metrics.tokens_per_second == 0.0
        assert metrics.memory_usage_mb == 0.0
        assert metrics.cpu_usage_percent == 0.0

    def test_performance_metrics_with_values(self):
        """Test performance metrics with specific values."""
        metrics = RAGPerformanceMetrics(
            embedding_latency_ms=150.0,
            retrieval_latency_ms=200.0,
            synthesis_latency_ms=800.0,
            total_latency_ms=1150.0,
            tokens_per_second=25.5,
            memory_usage_mb=512.0,
            cpu_usage_percent=75.2,
        )

        assert metrics.embedding_latency_ms == 150.0
        assert metrics.retrieval_latency_ms == 200.0
        assert metrics.synthesis_latency_ms == 800.0
        assert metrics.total_latency_ms == 1150.0
        assert metrics.tokens_per_second == 25.5
        assert metrics.memory_usage_mb == 512.0
        assert metrics.cpu_usage_percent == 75.2


class TestRAGOperationAnalytics:
    """Test RAG operation analytics dataclass."""

    def test_operation_analytics_initialization(self):
        """Test operation analytics initialization with default values."""
        analytics = RAGOperationAnalytics()

        assert analytics.total_operations == 0
        assert analytics.avg_cost_per_query == 0.0
        assert analytics.avg_response_time_ms == 0.0
        assert analytics.embedding_success_rate == 1.0
        assert analytics.retrieval_success_rate == 1.0
        assert analytics.synthesis_success_rate == 1.0
        assert analytics.avg_retrieval_relevance is None
        assert analytics.recommendations == []

    def test_operation_analytics_with_values(self):
        """Test operation analytics with specific values."""
        recommendations = ["Use smaller embedding model", "Enable caching"]

        analytics = RAGOperationAnalytics(
            total_operations=100,
            avg_cost_per_query=0.005,
            avg_response_time_ms=1250.0,
            embedding_success_rate=0.98,
            retrieval_success_rate=0.95,
            synthesis_success_rate=0.97,
            avg_retrieval_relevance=0.82,
            recommendations=recommendations,
        )

        assert analytics.total_operations == 100
        assert analytics.avg_cost_per_query == 0.005
        assert analytics.avg_response_time_ms == 1250.0
        assert analytics.embedding_success_rate == 0.98
        assert analytics.retrieval_success_rate == 0.95
        assert analytics.synthesis_success_rate == 0.97
        assert analytics.avg_retrieval_relevance == 0.82
        assert analytics.recommendations == recommendations


class TestRAGOperationMonitoring:
    """Test RAG operation monitoring functionality."""

    @pytest.fixture
    def monitor(self):
        """Create RAG monitor for testing."""
        return RAGMonitor(
            enable_quality_metrics=True,
            enable_cost_tracking=True,
            enable_performance_profiling=True,
        )

    def test_monitor_rag_operation_context_manager(self, monitor):
        """Test monitor_rag_operation context manager basic usage."""
        with monitor.monitor_rag_operation("test query") as operation_context:
            assert operation_context is not None
            assert hasattr(operation_context, "query")
            assert operation_context.query == "test query"
            assert operation_context.start_time is not None

    def test_monitor_rag_operation_with_governance_attrs(self, monitor):
        """Test monitor_rag_operation with governance attributes."""
        with monitor.monitor_rag_operation(
            "test query", team="test-team", project="test-project", complexity="high"
        ) as operation_context:
            assert operation_context.governance_attrs["team"] == "test-team"
            assert operation_context.governance_attrs["project"] == "test-project"
            assert operation_context.governance_attrs["complexity"] == "high"

    def test_monitor_rag_operation_timing(self, monitor):
        """Test that operation timing is recorded correctly."""
        with monitor.monitor_rag_operation("timing test"):
            # Simulate some processing time
            time.sleep(0.1)

        # Check that the operation was recorded
        assert len(monitor.operations) == 1
        operation = monitor.operations[0]

        assert operation["query"] == "timing test"
        assert operation["end_time"] > operation["start_time"]
        assert operation["duration_ms"] >= 100  # At least 100ms

    def test_monitor_rag_operation_exception_handling(self, monitor):
        """Test that exceptions are handled properly in monitoring."""
        with pytest.raises(ValueError):
            with monitor.monitor_rag_operation("error test"):
                raise ValueError("Test exception")

        # Operation should still be recorded even with exception
        assert len(monitor.operations) == 1
        operation = monitor.operations[0]
        assert operation["success"] is False
        assert "error" in operation

    def test_multiple_concurrent_operations(self, monitor):
        """Test multiple concurrent operations."""
        import threading

        def run_operation(operation_id):
            with monitor.monitor_rag_operation(f"concurrent query {operation_id}"):
                time.sleep(0.05)  # Simulate processing

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=run_operation, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All operations should be recorded
        assert len(monitor.operations) == 3
        queries = [op["query"] for op in monitor.operations]
        assert "concurrent query 0" in queries
        assert "concurrent query 1" in queries
        assert "concurrent query 2" in queries


class TestQualityMetricsTracking:
    """Test quality metrics tracking functionality."""

    @pytest.fixture
    def monitor(self):
        """Create RAG monitor with quality metrics enabled."""
        return RAGMonitor(enable_quality_metrics=True)

    def test_record_quality_metrics(self, monitor):
        """Test recording quality metrics."""
        metrics = RAGQualityMetrics(
            retrieval_relevance=0.85, response_faithfulness=0.90, answer_relevancy=0.88
        )

        monitor.record_quality_metrics("test_operation", metrics)

        assert len(monitor.quality_scores) == 1
        recorded_metrics = monitor.quality_scores[0]
        assert recorded_metrics["operation_id"] == "test_operation"
        assert recorded_metrics["metrics"].retrieval_relevance == 0.85
        assert recorded_metrics["metrics"].response_faithfulness == 0.90
        assert recorded_metrics["metrics"].answer_relevancy == 0.88

    def test_quality_metrics_disabled(self):
        """Test behavior when quality metrics are disabled."""
        monitor = RAGMonitor(enable_quality_metrics=False)

        metrics = RAGQualityMetrics(retrieval_relevance=0.85)
        monitor.record_quality_metrics("test_operation", metrics)

        # Should not record metrics when disabled
        assert len(monitor.quality_scores) == 0

    def test_calculate_average_quality_metrics(self, monitor):
        """Test calculation of average quality metrics."""
        # Record multiple quality measurements
        metrics1 = RAGQualityMetrics(
            retrieval_relevance=0.8, response_faithfulness=0.9, answer_relevancy=0.85
        )
        metrics2 = RAGQualityMetrics(
            retrieval_relevance=0.9, response_faithfulness=0.8, answer_relevancy=0.75
        )

        monitor.record_quality_metrics("op1", metrics1)
        monitor.record_quality_metrics("op2", metrics2)

        avg_metrics = monitor.calculate_average_quality_metrics()

        # Check averages
        assert avg_metrics.retrieval_relevance == 0.85  # (0.8 + 0.9) / 2
        assert avg_metrics.response_faithfulness == 0.85  # (0.9 + 0.8) / 2
        assert avg_metrics.answer_relevancy == 0.8  # (0.85 + 0.75) / 2

    def test_calculate_average_quality_metrics_empty(self, monitor):
        """Test calculation of average quality metrics with no data."""
        avg_metrics = monitor.calculate_average_quality_metrics()

        # Should return default metrics when no data
        assert avg_metrics.retrieval_relevance == 0.0
        assert avg_metrics.response_faithfulness == 0.0
        assert avg_metrics.answer_relevancy == 0.0


class TestPerformanceMetricsTracking:
    """Test performance metrics tracking functionality."""

    @pytest.fixture
    def monitor(self):
        """Create RAG monitor with performance profiling enabled."""
        return RAGMonitor(enable_performance_profiling=True)

    def test_record_performance_metrics(self, monitor):
        """Test recording performance metrics."""
        metrics = RAGPerformanceMetrics(
            embedding_latency_ms=150.0,
            retrieval_latency_ms=200.0,
            synthesis_latency_ms=800.0,
            total_latency_ms=1150.0,
            tokens_per_second=25.5,
        )

        monitor.record_performance_metrics("test_operation", metrics)

        assert len(monitor.performance_metrics) == 1
        recorded_metrics = monitor.performance_metrics[0]
        assert recorded_metrics["operation_id"] == "test_operation"
        assert recorded_metrics["metrics"].embedding_latency_ms == 150.0
        assert recorded_metrics["metrics"].retrieval_latency_ms == 200.0
        assert recorded_metrics["metrics"].synthesis_latency_ms == 800.0
        assert recorded_metrics["metrics"].total_latency_ms == 1150.0
        assert recorded_metrics["metrics"].tokens_per_second == 25.5

    def test_performance_metrics_disabled(self):
        """Test behavior when performance metrics are disabled."""
        monitor = RAGMonitor(enable_performance_profiling=False)

        metrics = RAGPerformanceMetrics(total_latency_ms=1000.0)
        monitor.record_performance_metrics("test_operation", metrics)

        # Should not record metrics when disabled
        assert len(monitor.performance_metrics) == 0

    def test_calculate_average_performance_metrics(self, monitor):
        """Test calculation of average performance metrics."""
        metrics1 = RAGPerformanceMetrics(
            embedding_latency_ms=100.0,
            retrieval_latency_ms=200.0,
            synthesis_latency_ms=600.0,
            total_latency_ms=900.0,
        )
        metrics2 = RAGPerformanceMetrics(
            embedding_latency_ms=200.0,
            retrieval_latency_ms=300.0,
            synthesis_latency_ms=800.0,
            total_latency_ms=1300.0,
        )

        monitor.record_performance_metrics("op1", metrics1)
        monitor.record_performance_metrics("op2", metrics2)

        avg_metrics = monitor.calculate_average_performance_metrics()

        # Check averages
        assert avg_metrics.embedding_latency_ms == 150.0  # (100 + 200) / 2
        assert avg_metrics.retrieval_latency_ms == 250.0  # (200 + 300) / 2
        assert avg_metrics.synthesis_latency_ms == 700.0  # (600 + 800) / 2
        assert avg_metrics.total_latency_ms == 1100.0  # (900 + 1300) / 2


class TestRAGAnalytics:
    """Test RAG analytics generation."""

    @pytest.fixture
    def monitor_with_data(self):
        """Create RAG monitor with sample data."""
        monitor = RAGMonitor()

        # Add sample operations
        with monitor.monitor_rag_operation("query 1", cost=0.005):
            time.sleep(0.01)

        with monitor.monitor_rag_operation("query 2", cost=0.008):
            time.sleep(0.02)

        with monitor.monitor_rag_operation("query 3", cost=0.003):
            time.sleep(0.01)

        # Add quality metrics
        monitor.record_quality_metrics(
            "query_1",
            RAGQualityMetrics(retrieval_relevance=0.85, response_faithfulness=0.90),
        )
        monitor.record_quality_metrics(
            "query_2",
            RAGQualityMetrics(retrieval_relevance=0.80, response_faithfulness=0.85),
        )

        # Add performance metrics
        monitor.record_performance_metrics(
            "query_1",
            RAGPerformanceMetrics(total_latency_ms=1200.0, tokens_per_second=20.0),
        )
        monitor.record_performance_metrics(
            "query_2",
            RAGPerformanceMetrics(total_latency_ms=1500.0, tokens_per_second=15.0),
        )

        return monitor

    def test_get_analytics(self, monitor_with_data):
        """Test getting analytics from monitor."""
        analytics = monitor_with_data.get_analytics()

        assert isinstance(analytics, RAGOperationAnalytics)
        assert analytics.total_operations == 3
        assert analytics.avg_cost_per_query > 0.0
        assert analytics.avg_response_time_ms > 0.0

        # Quality metrics should be available
        assert analytics.avg_retrieval_relevance is not None
        assert analytics.avg_retrieval_relevance > 0.8

    def test_analytics_with_no_data(self):
        """Test analytics with no recorded data."""
        monitor = RAGMonitor()
        analytics = monitor.get_analytics()

        assert analytics.total_operations == 0
        assert analytics.avg_cost_per_query == 0.0
        assert analytics.avg_response_time_ms == 0.0
        assert analytics.avg_retrieval_relevance is None
        assert analytics.recommendations == []

    def test_generate_recommendations(self, monitor_with_data):
        """Test recommendation generation."""
        analytics = monitor_with_data.get_analytics()

        # Should have some recommendations based on the data
        assert isinstance(analytics.recommendations, list)

    def test_analytics_success_rates(self, monitor_with_data):
        """Test success rate calculations in analytics."""
        # Add a failed operation
        try:
            with monitor_with_data.monitor_rag_operation("failed query"):
                raise Exception("Simulated failure")
        except Exception:
            pass

        analytics = monitor_with_data.get_analytics()

        # Success rates should be calculated correctly
        # 3 successful out of 4 total = 75%
        assert analytics.embedding_success_rate == 0.75
        assert analytics.retrieval_success_rate == 0.75
        assert analytics.synthesis_success_rate == 0.75


class TestRAGMonitorFactory:
    """Test RAG monitor factory function."""

    def test_create_rag_monitor_default(self):
        """Test creating RAG monitor with default settings."""
        monitor = create_rag_monitor()

        assert isinstance(monitor, RAGMonitor)
        assert monitor.enable_quality_metrics is True
        assert monitor.enable_cost_tracking is True
        assert monitor.enable_performance_profiling is True

    def test_create_rag_monitor_custom_settings(self):
        """Test creating RAG monitor with custom settings."""
        monitor = create_rag_monitor(
            enable_quality_metrics=False,
            enable_cost_tracking=True,
            enable_performance_profiling=False,
            team="custom-team",
        )

        assert monitor.enable_quality_metrics is False
        assert monitor.enable_cost_tracking is True
        assert monitor.enable_performance_profiling is False
        assert monitor.default_governance_attrs["team"] == "custom-team"

    def test_create_rag_monitor_with_governance_attrs(self):
        """Test creating RAG monitor with governance attributes."""
        monitor = create_rag_monitor(
            team="analytics-team",
            project="rag-optimization",
            customer_id="enterprise-client",
            environment="production",
        )

        expected_attrs = {
            "team": "analytics-team",
            "project": "rag-optimization",
            "customer_id": "enterprise-client",
            "environment": "production",
        }

        assert monitor.default_governance_attrs == expected_attrs


class TestRAGMonitorEdgeCases:
    """Test edge cases and error conditions."""

    def test_monitor_with_very_short_operations(self):
        """Test monitoring very short operations."""
        monitor = RAGMonitor()

        with monitor.monitor_rag_operation("short operation"):
            pass  # No processing time

        assert len(monitor.operations) == 1
        operation = monitor.operations[0]
        assert operation["duration_ms"] >= 0  # Should handle very short durations

    def test_monitor_with_very_long_operations(self):
        """Test monitoring very long operations."""
        monitor = RAGMonitor()

        with monitor.monitor_rag_operation("long operation"):
            time.sleep(0.5)  # 500ms operation

        assert len(monitor.operations) == 1
        operation = monitor.operations[0]
        assert operation["duration_ms"] >= 500  # Should handle long durations

    def test_quality_metrics_with_invalid_scores(self):
        """Test quality metrics with invalid scores."""
        monitor = RAGMonitor()

        # Test with scores outside 0-1 range
        metrics = RAGQualityMetrics(
            retrieval_relevance=1.5,  # Invalid: > 1
            response_faithfulness=-0.1,  # Invalid: < 0
        )

        monitor.record_quality_metrics("invalid_test", metrics)

        # Should still record, but may clamp values in real implementation
        assert len(monitor.quality_scores) == 1

    def test_performance_metrics_with_negative_values(self):
        """Test performance metrics with negative values."""
        monitor = RAGMonitor()

        # Test with negative latency (which shouldn't happen in practice)
        metrics = RAGPerformanceMetrics(total_latency_ms=-100.0, tokens_per_second=-5.0)

        monitor.record_performance_metrics("negative_test", metrics)

        # Should still record, but may validate values in real implementation
        assert len(monitor.performance_metrics) == 1

    def test_monitor_memory_usage_with_many_operations(self):
        """Test monitor memory usage with many operations."""
        monitor = RAGMonitor()

        # Add many operations to test memory management
        for i in range(1000):
            with monitor.monitor_rag_operation(f"operation {i}"):
                pass

        assert len(monitor.operations) == 1000

        # In a real implementation, might implement circular buffer
        # or cleanup strategies for memory management


if __name__ == "__main__":
    pytest.main([__file__])
