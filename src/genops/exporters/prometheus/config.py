"""Configuration management for Prometheus exporter."""

import os
from dataclasses import dataclass, field


@dataclass
class PrometheusConfig:
    """Configuration for Prometheus metrics exporter.

    Attributes:
        port: Port for metrics endpoint (default: 8000)
        metrics_path: Path for metrics endpoint (default: /metrics)
        namespace: Metrics namespace prefix (default: genops)
        prometheus_url: URL of Prometheus server for validation (default: http://localhost:9090)
        scrape_interval: Expected Prometheus scrape interval in seconds (default: 15)
        enable_recording_rules: Enable recording rules templates (default: True)
        enable_alert_rules: Enable alert rules templates (default: True)
        max_label_cardinality: Maximum unique label combinations (default: 10000)
        sampling_rate: Sampling rate for high-volume scenarios (default: 1.0 = 100%)
        include_labels: Specific labels to include (empty = include all)
        exclude_labels: Specific labels to exclude
    """

    port: int = 8000
    metrics_path: str = "/metrics"
    namespace: str = "genops"
    prometheus_url: str = "http://localhost:9090"
    scrape_interval: int = 15
    enable_recording_rules: bool = True
    enable_alert_rules: bool = True
    max_label_cardinality: int = 10000
    sampling_rate: float = 1.0
    include_labels: set[str] = field(default_factory=set)
    exclude_labels: set[str] = field(default_factory=set)

    @classmethod
    def from_env(cls) -> "PrometheusConfig":
        """Load configuration from environment variables.

        Environment variables:
            PROMETHEUS_EXPORTER_PORT: Metrics endpoint port
            PROMETHEUS_METRICS_PATH: Metrics endpoint path
            PROMETHEUS_NAMESPACE: Metrics namespace prefix
            PROMETHEUS_URL: Prometheus server URL
            PROMETHEUS_SCRAPE_INTERVAL: Scrape interval in seconds
            PROMETHEUS_MAX_CARDINALITY: Maximum label cardinality
            PROMETHEUS_SAMPLING_RATE: Sampling rate (0.0-1.0)
            PROMETHEUS_INCLUDE_LABELS: Comma-separated labels to include
            PROMETHEUS_EXCLUDE_LABELS: Comma-separated labels to exclude

        Returns:
            PrometheusConfig instance with environment overrides
        """
        config = cls()

        # Port configuration
        if port_str := os.getenv("PROMETHEUS_EXPORTER_PORT"):
            try:
                config.port = int(port_str)
            except ValueError:
                pass

        # Path configuration
        if metrics_path := os.getenv("PROMETHEUS_METRICS_PATH"):
            config.metrics_path = metrics_path

        # Namespace configuration
        if namespace := os.getenv("PROMETHEUS_NAMESPACE"):
            config.namespace = namespace

        # Prometheus URL
        if prometheus_url := os.getenv("PROMETHEUS_URL"):
            config.prometheus_url = prometheus_url

        # Scrape interval
        if scrape_str := os.getenv("PROMETHEUS_SCRAPE_INTERVAL"):
            try:
                config.scrape_interval = int(scrape_str)
            except ValueError:
                pass

        # Max cardinality
        if cardinality_str := os.getenv("PROMETHEUS_MAX_CARDINALITY"):
            try:
                config.max_label_cardinality = int(cardinality_str)
            except ValueError:
                pass

        # Sampling rate
        if sampling_str := os.getenv("PROMETHEUS_SAMPLING_RATE"):
            try:
                sampling_rate = float(sampling_str)
                if 0.0 <= sampling_rate <= 1.0:
                    config.sampling_rate = sampling_rate
            except ValueError:
                pass

        # Label filtering
        if include_labels := os.getenv("PROMETHEUS_INCLUDE_LABELS"):
            config.include_labels = {
                label.strip() for label in include_labels.split(",") if label.strip()
            }

        if exclude_labels := os.getenv("PROMETHEUS_EXCLUDE_LABELS"):
            config.exclude_labels = {
                label.strip() for label in exclude_labels.split(",") if label.strip()
            }

        return config

    def validate(self) -> tuple[bool, list[str]]:
        """Validate configuration settings.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Validate port range
        if not (1024 <= self.port <= 65535):
            errors.append(f"Port {self.port} outside valid range (1024-65535)")

        # Validate sampling rate
        if not (0.0 <= self.sampling_rate <= 1.0):
            errors.append(
                f"Sampling rate {self.sampling_rate} must be between 0.0 and 1.0"
            )

        # Validate scrape interval
        if self.scrape_interval <= 0:
            errors.append(f"Scrape interval {self.scrape_interval} must be positive")

        # Validate max cardinality
        if self.max_label_cardinality <= 0:
            errors.append(
                f"Max label cardinality {self.max_label_cardinality} must be positive"
            )

        # Validate namespace (must be valid Prometheus metric name prefix)
        if not self.namespace.replace("_", "").isalnum():
            errors.append(
                f"Namespace '{self.namespace}' contains invalid characters (use alphanumeric and underscores only)"
            )

        return (len(errors) == 0, errors)
