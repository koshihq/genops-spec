"""Resource monitoring for Ollama local model deployments."""

import logging
import threading
import time

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None  # type: ignore[assignment]
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Try to import GPU monitoring libraries
try:
    import GPUtil

    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

try:
    import pynvml

    pynvml.nvmlInit()
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False
except Exception as e:
    logger.warning(f"Failed to initialize NVIDIA ML: {e}")
    HAS_PYNVML = False


@dataclass
class ResourceMetrics:
    """Real-time resource utilization metrics."""

    timestamp: float

    # CPU metrics
    cpu_usage_percent: float = 0.0
    cpu_temperature: Optional[float] = None

    # Memory metrics
    memory_usage_mb: float = 0.0
    memory_available_mb: float = 0.0
    memory_percent: float = 0.0

    # GPU metrics (if available)
    gpu_usage_percent: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_temperature: Optional[float] = None
    gpu_power_draw_watts: Optional[float] = None

    # System metrics
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0


@dataclass
class HardwareMetrics:
    """Hardware utilization summary over time."""

    measurement_count: int = 0
    duration_seconds: float = 0.0

    # CPU statistics
    avg_cpu_usage: float = 0.0
    max_cpu_usage: float = 0.0
    cpu_hours: float = 0.0

    # Memory statistics
    avg_memory_usage_mb: float = 0.0
    max_memory_usage_mb: float = 0.0

    # GPU statistics
    avg_gpu_usage: float = 0.0
    max_gpu_usage: float = 0.0
    avg_gpu_memory_mb: float = 0.0
    max_gpu_memory_mb: float = 0.0
    gpu_hours: float = 0.0

    # Efficiency metrics
    tokens_per_gpu_hour: float = 0.0
    cost_per_gpu_hour: float = 0.0
    energy_efficiency_score: float = 0.0


@dataclass
class ModelPerformanceTracker:
    """Tracks performance metrics for specific models."""

    model_name: str
    total_inferences: int = 0
    total_inference_time_ms: float = 0.0

    # Performance statistics
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Resource utilization during inference
    avg_gpu_utilization: float = 0.0
    avg_memory_usage_mb: float = 0.0

    # Token throughput
    total_tokens: int = 0
    avg_tokens_per_second: float = 0.0

    # Efficiency metrics
    tokens_per_gpu_hour: float = 0.0
    inferences_per_dollar: float = 0.0

    # Latency history (for percentile calculations)
    latency_history: deque = field(default_factory=lambda: deque(maxlen=1000))

    def add_inference(
        self,
        latency_ms: float,
        tokens: int = 0,
        gpu_utilization: float = 0.0,
        memory_mb: float = 0.0,
    ):
        """Add a new inference measurement."""
        self.total_inferences += 1
        self.total_inference_time_ms += latency_ms
        self.total_tokens += tokens

        # Update averages
        self.avg_latency_ms = self.total_inference_time_ms / self.total_inferences
        if tokens > 0 and latency_ms > 0:
            tokens_per_second = tokens / (latency_ms / 1000)
            self.avg_tokens_per_second = (
                self.avg_tokens_per_second * (self.total_inferences - 1)
                + tokens_per_second
            ) / self.total_inferences

        # Update resource utilization
        if gpu_utilization > 0:
            self.avg_gpu_utilization = (
                self.avg_gpu_utilization * (self.total_inferences - 1) + gpu_utilization
            ) / self.total_inferences

        if memory_mb > 0:
            self.avg_memory_usage_mb = (
                self.avg_memory_usage_mb * (self.total_inferences - 1) + memory_mb
            ) / self.total_inferences

        # Add to latency history for percentile calculations
        self.latency_history.append(latency_ms)

        # Update percentiles
        self._update_percentiles()

    def _update_percentiles(self):
        """Update latency percentiles from history."""
        if not self.latency_history:
            return

        sorted_latencies = sorted(self.latency_history)
        n = len(sorted_latencies)

        self.p50_latency_ms = sorted_latencies[int(n * 0.50)]
        self.p95_latency_ms = sorted_latencies[int(n * 0.95)]
        self.p99_latency_ms = sorted_latencies[int(n * 0.99)]


class OllamaResourceMonitor:
    """
    Comprehensive resource monitoring for Ollama deployments.

    Tracks:
    - Real-time CPU, GPU, and memory utilization
    - Model-specific performance metrics
    - Infrastructure cost attribution
    - Resource optimization recommendations
    """

    def __init__(
        self,
        monitoring_interval: float = 1.0,
        history_size: int = 1000,
        enable_gpu_monitoring: bool = True,
        enable_detailed_metrics: bool = True,
    ):
        """
        Initialize resource monitor.

        Args:
            monitoring_interval: Seconds between resource measurements
            history_size: Number of historical measurements to keep
            enable_gpu_monitoring: Enable GPU utilization tracking
            enable_detailed_metrics: Enable detailed performance metrics
        """
        if not HAS_PSUTIL:
            raise ImportError(
                "psutil is required for OllamaResourceMonitor. "
                "Install with: pip install psutil"
            )

        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_gpu_monitoring = enable_gpu_monitoring and (
            HAS_GPUTIL or HAS_PYNVML
        )
        self.enable_detailed_metrics = enable_detailed_metrics

        # Resource history
        self.resource_history: deque = deque(maxlen=history_size)

        # Model performance tracking
        self.model_trackers: dict[str, ModelPerformanceTracker] = {}

        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Hardware info
        self.hardware_info = self._get_hardware_info()

        logger.info(
            f"Initialized Ollama resource monitor (GPU monitoring: {self.enable_gpu_monitoring})"
        )

    def _get_hardware_info(self) -> dict[str, Any]:
        """Get static hardware information."""
        info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "gpu_available": self.enable_gpu_monitoring,
        }

        if self.enable_gpu_monitoring and HAS_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    info.update(
                        {
                            "gpu_name": gpu.name,
                            "gpu_memory_gb": gpu.memoryTotal / 1024,
                            "gpu_driver_version": gpu.driver,
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to get GPU info: {e}")

        return info

    def start_monitoring(self):
        """Start background resource monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitor_thread.start()
        logger.info("Started resource monitoring")

    def stop_monitoring(self):
        """Stop background resource monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Stopped resource monitoring")

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_resource_metrics()
                self.resource_history.append(metrics)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.monitoring_interval)

    def _collect_resource_metrics(self) -> ResourceMetrics:
        """Collect current resource utilization metrics."""
        metrics = ResourceMetrics(timestamp=time.time())

        # CPU metrics
        metrics.cpu_usage_percent = psutil.cpu_percent(interval=None)

        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.memory_usage_mb = (memory.total - memory.available) / (1024**2)
        metrics.memory_available_mb = memory.available / (1024**2)
        metrics.memory_percent = memory.percent

        # GPU metrics
        if self.enable_gpu_monitoring:
            try:
                if HAS_PYNVML:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

                    # GPU utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics.gpu_usage_percent = util.gpu

                    # GPU memory
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    metrics.gpu_memory_used_mb = mem_info.used / (1024**2)
                    metrics.gpu_memory_total_mb = mem_info.total / (1024**2)

                    # GPU temperature and power
                    try:
                        metrics.gpu_temperature = pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                        metrics.gpu_power_draw_watts = (
                            pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                        )
                    except Exception:
                        pass  # These might not be available on all cards

                elif HAS_GPUTIL:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        metrics.gpu_usage_percent = gpu.load * 100
                        metrics.gpu_memory_used_mb = gpu.memoryUsed
                        metrics.gpu_memory_total_mb = gpu.memoryTotal
                        metrics.gpu_temperature = gpu.temperature

            except Exception as e:
                logger.debug(f"Failed to collect GPU metrics: {e}")

        return metrics

    @contextmanager
    def monitor_inference(self, model_name: str, operation_id: str = None):  # type: ignore[assignment]
        """
        Context manager to monitor a specific inference operation.

        Args:
            model_name: Name of the model being used
            operation_id: Optional operation identifier

        Yields:
            Dictionary to store inference results
        """
        if model_name not in self.model_trackers:
            self.model_trackers[model_name] = ModelPerformanceTracker(
                model_name=model_name
            )

        tracker = self.model_trackers[model_name]

        # Start monitoring if not already running
        if not self.is_monitoring:
            self.start_monitoring()

        # Collect baseline metrics
        start_time = time.time()
        self._collect_resource_metrics()

        inference_data = {
            "start_time": start_time,
            "model_name": model_name,
            "operation_id": operation_id,
            "tokens": 0,
            "success": False,
        }

        try:
            yield inference_data
            inference_data["success"] = True

        finally:
            # Collect final metrics
            end_time = time.time()
            final_metrics = self._collect_resource_metrics()

            # Calculate inference duration and metrics
            duration_ms = (end_time - start_time) * 1000

            # Calculate average GPU utilization during inference
            gpu_utilization = final_metrics.gpu_usage_percent
            memory_usage = final_metrics.gpu_memory_used_mb

            # Update model tracker
            tracker.add_inference(
                latency_ms=duration_ms,
                tokens=inference_data.get("tokens", 0),  # type: ignore[arg-type]
                gpu_utilization=gpu_utilization,
                memory_mb=memory_usage,
            )

            # Log performance metrics
            if inference_data["success"]:
                logger.debug(
                    f"Inference completed: {model_name} - {duration_ms:.1f}ms, "
                    f"GPU: {gpu_utilization:.1f}%, Memory: {memory_usage:.1f}MB"
                )

    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get the most recent resource metrics."""
        if self.resource_history:
            return self.resource_history[-1]
        return self._collect_resource_metrics()

    def get_hardware_summary(self, duration_minutes: int = 60) -> HardwareMetrics:
        """
        Get hardware utilization summary over the specified duration.

        Args:
            duration_minutes: Duration to analyze in minutes

        Returns:
            Hardware utilization summary
        """
        cutoff_time = time.time() - (duration_minutes * 60)
        recent_metrics = [m for m in self.resource_history if m.timestamp > cutoff_time]

        if not recent_metrics:
            return HardwareMetrics()

        summary = HardwareMetrics(
            measurement_count=len(recent_metrics),
            duration_seconds=duration_minutes * 60,
        )

        # Calculate CPU statistics
        cpu_values = [m.cpu_usage_percent for m in recent_metrics]
        summary.avg_cpu_usage = sum(cpu_values) / len(cpu_values)
        summary.max_cpu_usage = max(cpu_values)
        summary.cpu_hours = (
            summary.duration_seconds / 3600 * (summary.avg_cpu_usage / 100)
        )

        # Calculate memory statistics
        memory_values = [m.memory_usage_mb for m in recent_metrics]
        summary.avg_memory_usage_mb = sum(memory_values) / len(memory_values)
        summary.max_memory_usage_mb = max(memory_values)

        # Calculate GPU statistics
        if self.enable_gpu_monitoring:
            gpu_values = [
                m.gpu_usage_percent for m in recent_metrics if m.gpu_usage_percent > 0
            ]
            gpu_memory_values = [
                m.gpu_memory_used_mb for m in recent_metrics if m.gpu_memory_used_mb > 0
            ]

            if gpu_values:
                summary.avg_gpu_usage = sum(gpu_values) / len(gpu_values)
                summary.max_gpu_usage = max(gpu_values)
                summary.gpu_hours = (
                    summary.duration_seconds / 3600 * (summary.avg_gpu_usage / 100)
                )

            if gpu_memory_values:
                summary.avg_gpu_memory_mb = sum(gpu_memory_values) / len(
                    gpu_memory_values
                )
                summary.max_gpu_memory_mb = max(gpu_memory_values)

        return summary

    def get_model_performance(
        self,
        model_name: str = None,  # type: ignore[assignment]
    ) -> dict[str, ModelPerformanceTracker]:
        """Get performance metrics for specific model or all models."""
        if model_name:
            return {model_name: self.model_trackers.get(model_name)}
        return self.model_trackers.copy()

    def get_optimization_recommendations(self) -> list[str]:
        """Generate resource optimization recommendations."""
        recommendations = []

        if not self.resource_history:
            return ["Start monitoring to get optimization recommendations"]

        current = self.get_current_metrics()
        hardware_summary = self.get_hardware_summary(duration_minutes=30)

        # CPU recommendations
        if hardware_summary.avg_cpu_usage > 80:
            recommendations.append(
                f"High CPU usage ({hardware_summary.avg_cpu_usage:.1f}%) - consider adding CPU cores or reducing concurrent requests"
            )
        elif hardware_summary.avg_cpu_usage < 20:
            recommendations.append(
                f"Low CPU usage ({hardware_summary.avg_cpu_usage:.1f}%) - you can handle more concurrent requests"
            )

        # GPU recommendations
        if self.enable_gpu_monitoring and hardware_summary.avg_gpu_usage > 0:
            if hardware_summary.avg_gpu_usage > 90:
                recommendations.append(
                    f"Very high GPU usage ({hardware_summary.avg_gpu_usage:.1f}%) - consider GPU scaling or model optimization"
                )
            elif hardware_summary.avg_gpu_usage < 30:
                recommendations.append(
                    f"Low GPU usage ({hardware_summary.avg_gpu_usage:.1f}%) - you can run larger models or more concurrent requests"
                )

            if (
                current
                and current.gpu_memory_used_mb / current.gpu_memory_total_mb > 0.9
            ):
                recommendations.append(
                    "GPU memory is >90% full - consider using smaller models or quantized versions"
                )

        # Memory recommendations
        if (
            hardware_summary.max_memory_usage_mb
            / (self.hardware_info["memory_total_gb"] * 1024)
            > 0.8
        ):
            recommendations.append(
                "High memory usage detected - consider adding RAM or optimizing model loading"
            )

        # Model-specific recommendations
        for model_name, tracker in self.model_trackers.items():
            if tracker.avg_latency_ms > 5000:  # >5 seconds
                recommendations.append(
                    f"Model '{model_name}' has high latency ({tracker.avg_latency_ms:.0f}ms) - consider using quantized version"
                )

            if tracker.avg_tokens_per_second < 10:
                recommendations.append(
                    f"Model '{model_name}' has low throughput ({tracker.avg_tokens_per_second:.1f} tokens/sec) - check GPU utilization"
                )

        return recommendations[:5]  # Limit to top 5 recommendations


# Global resource monitor instance
_global_monitor: Optional[OllamaResourceMonitor] = None


def get_resource_monitor() -> OllamaResourceMonitor:
    """Get or create global resource monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = OllamaResourceMonitor()
    return _global_monitor


def set_resource_monitor(monitor: OllamaResourceMonitor) -> None:
    """Set global resource monitor instance."""
    global _global_monitor
    _global_monitor = monitor


def create_resource_monitor(**kwargs) -> OllamaResourceMonitor:
    """Create a new resource monitor with specified configuration."""
    return OllamaResourceMonitor(**kwargs)


# Export main classes and functions
__all__ = [
    "OllamaResourceMonitor",
    "ResourceMetrics",
    "HardwareMetrics",
    "ModelPerformanceTracker",
    "get_resource_monitor",
    "set_resource_monitor",
    "create_resource_monitor",
]
