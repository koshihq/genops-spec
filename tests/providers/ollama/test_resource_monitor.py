"""Tests for Ollama resource monitor functionality."""

import time
from unittest.mock import Mock, patch

import pytest

from genops.providers.ollama.resource_monitor import (
    HardwareMetrics,
    ModelPerformanceTracker,
    OllamaResourceMonitor,
    ResourceMetrics,
    create_resource_monitor,
    get_resource_monitor,
    set_resource_monitor,
)


class TestResourceMetrics:
    """Test ResourceMetrics dataclass."""

    def test_metrics_creation(self):
        """Test basic metrics creation."""
        timestamp = time.time()
        metrics = ResourceMetrics(
            timestamp=timestamp,
            cpu_usage_percent=50.0,
            memory_usage_mb=8192.0,
            gpu_usage_percent=75.0
        )

        assert metrics.timestamp == timestamp
        assert metrics.cpu_usage_percent == 50.0
        assert metrics.memory_usage_mb == 8192.0
        assert metrics.gpu_usage_percent == 75.0

        # Test defaults
        assert metrics.cpu_temperature is None
        assert metrics.gpu_power_draw_watts is None


class TestHardwareMetrics:
    """Test HardwareMetrics dataclass."""

    def test_hardware_metrics_creation(self):
        """Test hardware metrics creation with defaults."""
        metrics = HardwareMetrics()

        assert metrics.measurement_count == 0
        assert metrics.duration_seconds == 0.0
        assert metrics.avg_cpu_usage == 0.0
        assert metrics.max_cpu_usage == 0.0
        assert metrics.energy_efficiency_score == 0.0


class TestModelPerformanceTracker:
    """Test ModelPerformanceTracker functionality."""

    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = ModelPerformanceTracker(model_name="llama3.2:1b")

        assert tracker.model_name == "llama3.2:1b"
        assert tracker.total_inferences == 0
        assert tracker.avg_latency_ms == 0.0
        assert len(tracker.latency_history) == 0

    def test_add_inference_basic(self):
        """Test adding basic inference data."""
        tracker = ModelPerformanceTracker(model_name="test-model")

        tracker.add_inference(latency_ms=1500.0, tokens=50, gpu_utilization=80.0)

        assert tracker.total_inferences == 1
        assert tracker.avg_latency_ms == 1500.0
        assert tracker.total_tokens == 50
        assert tracker.avg_gpu_utilization == 80.0
        assert len(tracker.latency_history) == 1

    def test_add_multiple_inferences(self):
        """Test adding multiple inferences and averaging."""
        tracker = ModelPerformanceTracker(model_name="test-model")

        # Add multiple inferences
        tracker.add_inference(latency_ms=1000.0, tokens=30, gpu_utilization=70.0)
        tracker.add_inference(latency_ms=2000.0, tokens=40, gpu_utilization=80.0)
        tracker.add_inference(latency_ms=1500.0, tokens=35, gpu_utilization=75.0)

        assert tracker.total_inferences == 3
        assert tracker.avg_latency_ms == 1500.0  # (1000 + 2000 + 1500) / 3
        assert tracker.total_tokens == 105  # 30 + 40 + 35
        assert tracker.avg_gpu_utilization == 75.0  # (70 + 80 + 75) / 3

    def test_tokens_per_second_calculation(self):
        """Test tokens per second calculation."""
        tracker = ModelPerformanceTracker(model_name="test-model")

        # 50 tokens in 2000ms = 25 tokens/second
        tracker.add_inference(latency_ms=2000.0, tokens=50)
        assert tracker.avg_tokens_per_second == 25.0

    def test_percentile_calculations(self):
        """Test latency percentile calculations."""
        tracker = ModelPerformanceTracker(model_name="test-model")

        # Add various latencies
        latencies = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        for latency in latencies:
            tracker.add_inference(latency_ms=float(latency))

        # Check percentiles are reasonable
        assert tracker.p50_latency_ms == 500.0  # Middle value
        assert tracker.p95_latency_ms == 950.0   # 95th percentile
        assert tracker.p99_latency_ms == 990.0   # 99th percentile

    def test_latency_history_maxlen(self):
        """Test that latency history respects maxlen."""
        tracker = ModelPerformanceTracker(model_name="test-model")

        # Add more than maxlen (1000) entries
        for i in range(1200):
            tracker.add_inference(latency_ms=float(i))

        # Should only keep last 1000
        assert len(tracker.latency_history) == 1000
        assert tracker.latency_history[0] == 200.0  # First kept entry
        assert tracker.latency_history[-1] == 1199.0  # Last entry


class TestOllamaResourceMonitor:
    """Test OllamaResourceMonitor functionality."""

    @pytest.fixture
    def mock_psutil(self):
        """Mock psutil for system resource monitoring."""
        with patch('genops.providers.ollama.resource_monitor.psutil') as mock_ps:
            # Mock CPU
            mock_ps.cpu_count.return_value = 8
            mock_ps.cpu_percent.return_value = 45.0

            # Mock memory
            mock_memory = Mock()
            mock_memory.total = 16 * 1024**3  # 16GB
            mock_memory.available = 8 * 1024**3  # 8GB
            mock_memory.percent = 50.0
            mock_ps.virtual_memory.return_value = mock_memory

            yield mock_ps

    @pytest.fixture
    def mock_gputil(self):
        """Mock GPUtil for GPU monitoring."""
        with patch('genops.providers.ollama.resource_monitor.GPUtil') as mock_gpu:
            mock_gpu_device = Mock()
            mock_gpu_device.name = "NVIDIA RTX 4090"
            mock_gpu_device.memoryTotal = 24576  # 24GB
            mock_gpu_device.load = 0.8  # 80%
            mock_gpu_device.memoryUsed = 16384  # 16GB
            mock_gpu_device.temperature = 75
            mock_gpu_device.driver = "525.89"

            mock_gpu.getGPUs.return_value = [mock_gpu_device]
            yield mock_gpu

    @pytest.fixture
    def monitor(self, mock_psutil):
        """Create monitor instance for testing."""
        with patch('genops.providers.ollama.resource_monitor.HAS_GPUTIL', False):
            return OllamaResourceMonitor(
                monitoring_interval=0.1,  # Fast for testing
                enable_gpu_monitoring=False  # Disable for basic tests
            )

    def test_monitor_initialization(self, mock_psutil):
        """Test monitor initialization."""
        monitor = OllamaResourceMonitor()

        assert monitor.monitoring_interval == 1.0
        assert monitor.history_size == 1000
        assert not monitor.is_monitoring
        assert len(monitor.resource_history) == 0
        assert len(monitor.model_trackers) == 0

    def test_hardware_info_collection(self, mock_psutil, mock_gputil):
        """Test hardware info collection."""
        with patch('genops.providers.ollama.resource_monitor.HAS_GPUTIL', True):
            monitor = OllamaResourceMonitor(enable_gpu_monitoring=True)

            info = monitor.hardware_info
            assert info["cpu_count"] == 8
            assert info["memory_total_gb"] == 16.0
            assert info["gpu_available"] is True
            assert "gpu_name" in info

    def test_resource_metrics_collection(self, monitor, mock_psutil):
        """Test resource metrics collection."""
        metrics = monitor._collect_resource_metrics()

        assert isinstance(metrics, ResourceMetrics)
        assert metrics.cpu_usage_percent == 45.0
        assert metrics.memory_usage_mb == 8192.0  # 8GB
        assert metrics.memory_percent == 50.0
        assert metrics.timestamp > 0

    def test_gpu_metrics_collection(self, mock_psutil, mock_gputil):
        """Test GPU metrics collection."""
        with patch('genops.providers.ollama.resource_monitor.HAS_GPUTIL', True):
            monitor = OllamaResourceMonitor(enable_gpu_monitoring=True)
            metrics = monitor._collect_resource_metrics()

            assert metrics.gpu_usage_percent == 80.0
            assert metrics.gpu_memory_used_mb == 16384.0

    def test_monitoring_start_stop(self, monitor):
        """Test starting and stopping monitoring."""
        assert not monitor.is_monitoring

        monitor.start_monitoring()
        assert monitor.is_monitoring
        assert monitor.monitor_thread is not None

        # Let it run briefly
        time.sleep(0.2)

        monitor.stop_monitoring()
        assert not monitor.is_monitoring

    def test_monitor_inference_context_manager(self, monitor):
        """Test inference monitoring context manager."""
        model_name = "test-model"

        with monitor.monitor_inference(model_name) as inference_data:
            inference_data["tokens"] = 25
            time.sleep(0.05)  # Small delay to simulate inference

        # Check that tracker was created and updated
        assert model_name in monitor.model_trackers
        tracker = monitor.model_trackers[model_name]
        assert tracker.total_inferences == 1
        assert tracker.total_tokens == 25
        assert tracker.avg_latency_ms > 0

    def test_inference_context_with_error(self, monitor):
        """Test inference monitoring with error."""
        model_name = "error-model"

        try:
            with monitor.monitor_inference(model_name) as inference_data:
                inference_data["tokens"] = 10
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Should still record the inference attempt
        assert model_name in monitor.model_trackers
        tracker = monitor.model_trackers[model_name]
        assert tracker.total_inferences == 1

    def test_hardware_summary_calculation(self, monitor, mock_psutil):
        """Test hardware utilization summary."""
        # Add some mock resource history
        current_time = time.time()
        for i in range(10):
            metrics = ResourceMetrics(
                timestamp=current_time - (i * 60),  # 1 minute intervals
                cpu_usage_percent=50.0 + i,
                memory_usage_mb=8000.0 + (i * 100),
                gpu_usage_percent=70.0 + i
            )
            monitor.resource_history.append(metrics)

        summary = monitor.get_hardware_summary(duration_minutes=15)

        assert summary.measurement_count == 10
        assert summary.avg_cpu_usage > 50.0
        assert summary.max_cpu_usage == 59.0  # 50 + 9
        assert summary.avg_memory_usage_mb > 8000.0

    def test_optimization_recommendations(self, monitor):
        """Test optimization recommendations generation."""
        # Add some resource history with high usage
        current_time = time.time()
        for i in range(5):
            metrics = ResourceMetrics(
                timestamp=current_time - (i * 60),
                cpu_usage_percent=85.0,  # High CPU usage
                memory_usage_mb=12000.0,
                gpu_usage_percent=95.0   # Very high GPU usage
            )
            monitor.resource_history.append(metrics)

        recommendations = monitor.get_optimization_recommendations()

        assert len(recommendations) > 0
        # Should recommend addressing high resource usage
        high_usage_recs = [r for r in recommendations if "high" in r.lower()]
        assert len(high_usage_recs) > 0

    def test_model_performance_tracking(self, monitor):
        """Test model performance tracking."""
        model_name = "performance-model"

        # Simulate multiple inferences
        for i in range(5):
            with monitor.monitor_inference(model_name) as inference_data:
                inference_data["tokens"] = 30 + i
                time.sleep(0.01)  # Small delay

        performance = monitor.get_model_performance(model_name)
        tracker = performance[model_name]

        assert tracker.total_inferences == 5
        assert tracker.total_tokens == 30 + 31 + 32 + 33 + 34  # Sum of tokens
        assert tracker.avg_latency_ms > 0

    def test_get_current_metrics(self, monitor, mock_psutil):
        """Test getting current metrics."""
        current = monitor.get_current_metrics()

        assert isinstance(current, ResourceMetrics)
        assert current.cpu_usage_percent == 45.0
        assert current.memory_usage_mb == 8192.0

    def test_model_recommendations_generation(self, monitor):
        """Test model-specific recommendations."""
        model_name = "slow-model"

        # Create a slow model tracker
        tracker = ModelPerformanceTracker(model_name=model_name)
        # Add slow inferences
        for _ in range(3):
            tracker.add_inference(latency_ms=8000.0, tokens=5)  # 8 seconds, low throughput

        monitor.model_trackers[model_name] = tracker

        recommendations = monitor.get_optimization_recommendations()

        # Should recommend optimizations for slow model
        latency_recs = [r for r in recommendations if "latency" in r.lower()]
        assert len(latency_recs) > 0


class TestGlobalMonitorFunctions:
    """Test global monitor management functions."""

    def test_get_resource_monitor_singleton(self):
        """Test global monitor singleton behavior."""
        # Reset global state
        set_resource_monitor(None)

        monitor1 = get_resource_monitor()
        monitor2 = get_resource_monitor()

        # Should be same instance
        assert monitor1 is monitor2

    def test_set_resource_monitor(self):
        """Test setting global monitor instance."""
        custom_monitor = create_resource_monitor(monitoring_interval=0.5)
        set_resource_monitor(custom_monitor)

        retrieved_monitor = get_resource_monitor()
        assert retrieved_monitor is custom_monitor
        assert retrieved_monitor.monitoring_interval == 0.5

    def test_create_resource_monitor_factory(self):
        """Test monitor factory function."""
        monitor = create_resource_monitor(
            monitoring_interval=2.0,
            history_size=500,
            enable_gpu_monitoring=False
        )

        assert monitor.monitoring_interval == 2.0
        assert monitor.history_size == 500
        assert not monitor.enable_gpu_monitoring


class TestResourceMonitorErrorHandling:
    """Test error handling in resource monitor."""

    def test_gpu_monitoring_graceful_failure(self):
        """Test graceful failure when GPU monitoring unavailable."""
        with patch('genops.providers.ollama.resource_monitor.HAS_GPUTIL', False):
            monitor = OllamaResourceMonitor(enable_gpu_monitoring=True)

            # Should not enable GPU monitoring
            assert not monitor.enable_gpu_monitoring

    def test_metrics_collection_with_exceptions(self, mock_psutil):
        """Test metrics collection handles exceptions gracefully."""
        mock_psutil.cpu_percent.side_effect = Exception("CPU error")

        monitor = OllamaResourceMonitor(enable_gpu_monitoring=False)

        # Should not raise exception, but may have default values
        metrics = monitor._collect_resource_metrics()
        assert isinstance(metrics, ResourceMetrics)

    def test_monitoring_loop_exception_handling(self, monitor):
        """Test monitoring loop continues after exceptions."""
        # Mock the collection method to raise exception once
        original_method = monitor._collect_resource_metrics
        call_count = 0

        def failing_method():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Simulated failure")
            return original_method()

        monitor._collect_resource_metrics = failing_method

        monitor.start_monitoring()
        time.sleep(0.3)  # Let it run and handle the error
        monitor.stop_monitoring()

        # Should have called the method multiple times despite the error
        assert call_count > 1


class TestResourceMonitorIntegration:
    """Integration tests for resource monitor."""

    @pytest.fixture
    def integration_monitor(self, mock_psutil):
        """Create monitor for integration testing."""
        return OllamaResourceMonitor(
            monitoring_interval=0.1,
            enable_gpu_monitoring=False,
            enable_detailed_metrics=True
        )

    def test_full_monitoring_workflow(self, integration_monitor):
        """Test complete monitoring workflow."""
        monitor = integration_monitor

        # Start monitoring
        monitor.start_monitoring()

        # Simulate some inferences
        with monitor.monitor_inference("model1") as inf:
            inf["tokens"] = 25
            time.sleep(0.05)

        with monitor.monitor_inference("model2") as inf:
            inf["tokens"] = 40
            time.sleep(0.03)

        # Let monitoring collect some data
        time.sleep(0.25)

        # Stop monitoring
        monitor.stop_monitoring()

        # Verify data collection
        assert len(monitor.resource_history) > 0
        assert len(monitor.model_trackers) == 2

        # Check summaries
        summary = monitor.get_hardware_summary(duration_minutes=1)
        assert summary.measurement_count > 0

        performance = monitor.get_model_performance()
        assert "model1" in performance
        assert "model2" in performance

        # Check recommendations
        recommendations = monitor.get_optimization_recommendations()
        assert isinstance(recommendations, list)


if __name__ == "__main__":
    pytest.main([__file__])
