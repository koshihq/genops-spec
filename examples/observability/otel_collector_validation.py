"""Validation utilities for OpenTelemetry Collector integration setup."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urlparse

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class OTelCollectorValidationResult:
    """Result of OTel Collector setup validation."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    collector_healthy: bool = False
    collector_version: Optional[str] = None
    otlp_http_accessible: bool = False
    otlp_grpc_accessible: bool = False
    grafana_accessible: bool = False
    tempo_accessible: bool = False
    loki_accessible: bool = False
    mimir_accessible: bool = False

    @property
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return len(self.warnings) > 0


def check_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """
    Check if a TCP port is open and accepting connections.

    Args:
        host: Hostname or IP address
        port: Port number
        timeout: Connection timeout in seconds

    Returns:
        True if port is open and accepting connections
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def validate_url_format(url: str) -> tuple[bool, Optional[str]]:
    """
    Validate URL format.

    Args:
        url: URL to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url:
        return False, "URL is empty"

    try:
        parsed = urlparse(url)
        if not parsed.scheme:
            return False, "URL missing scheme (http/https)"
        if parsed.scheme not in ["http", "https"]:
            return False, f"Invalid URL scheme: {parsed.scheme} (expected http or https)"
        if not parsed.netloc:
            return False, "URL missing domain"
        return True, None
    except Exception as e:
        return False, f"Invalid URL format: {str(e)}"


def validate_setup(
    collector_endpoint: Optional[str] = None,
    grafana_endpoint: Optional[str] = None,
    check_connectivity: bool = True,
    check_backends: bool = True,
) -> OTelCollectorValidationResult:
    """
    Validate OpenTelemetry Collector integration setup.

    This function performs comprehensive validation of your OTel Collector configuration:
    1. Environment variables (OTEL_EXPORTER_OTLP_ENDPOINT)
    2. OTel Collector health check
    3. OTLP endpoint accessibility (HTTP and gRPC)
    4. Backend services (Grafana, Tempo, Loki, Mimir)
    5. OpenTelemetry dependencies

    Args:
        collector_endpoint: OTel Collector OTLP endpoint (or from OTEL_EXPORTER_OTLP_ENDPOINT env var)
        grafana_endpoint: Grafana endpoint (default: http://localhost:3000)
        check_connectivity: Test endpoint connectivity
        check_backends: Test backend service accessibility

    Returns:
        OTelCollectorValidationResult with validation details

    Example:
        >>> result = validate_setup()
        >>> if result.valid:
        ...     print("Setup validated successfully!")
        >>> else:
        ...     for error in result.errors:
        ...         print(f"Error: {error}")
    """
    result = OTelCollectorValidationResult(valid=False)

    # Check if requests library is available
    if check_connectivity and not HAS_REQUESTS:
        result.errors.append("requests library not installed")
        result.recommendations.append(
            "Install requests: pip install requests\n"
            "Or skip connectivity check: validate_setup(check_connectivity=False)"
        )
        return result

    # 1. Check environment variables and defaults
    env_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    final_endpoint = collector_endpoint or env_endpoint or "http://localhost:4318"
    final_grafana = grafana_endpoint or "http://localhost:3000"

    # Validate endpoint URL
    url_valid, url_error = validate_url_format(final_endpoint)
    if not url_valid:
        result.errors.append(f"Invalid collector endpoint URL: {url_error}")
        result.recommendations.append(
            f"Current endpoint: {final_endpoint}\n"
            "Expected format: http://localhost:4318"
        )

    # If basic validation failed, return early
    if result.errors:
        return result

    # Extract host and port from endpoint
    try:
        parsed = urlparse(final_endpoint)
        collector_host = parsed.hostname or "localhost"
        collector_http_port = parsed.port or 4318
        collector_grpc_port = 4317  # Standard gRPC port
        collector_health_port = 13133  # Standard health check port
    except Exception as e:
        result.errors.append(f"Failed to parse collector endpoint: {str(e)}")
        return result

    # 2. Check OTel Collector health endpoint
    if check_connectivity and HAS_REQUESTS:
        try:
            health_url = f"http://{collector_host}:{collector_health_port}/"
            response = requests.get(health_url, timeout=5)

            if response.status_code == 200:
                result.collector_healthy = True
                try:
                    health_data = response.json()
                    result.collector_version = health_data.get("status", "Collector is healthy")
                except Exception:
                    result.collector_version = "Collector is healthy"
            else:
                result.errors.append(
                    f"Collector health check failed (HTTP {response.status_code})"
                )
                result.recommendations.append(
                    f"Health check URL: {health_url}\n"
                    "Ensure Docker containers are running:\n"
                    "  docker-compose -f docker-compose.observability.yml ps"
                )

        except requests.exceptions.ConnectionError:
            result.errors.append("Collector not accessible - connection refused")
            result.recommendations.append(
                "Start the observability stack:\n"
                "  docker-compose -f docker-compose.observability.yml up -d\n"
                "\n"
                "Verify containers are running:\n"
                "  docker-compose -f docker-compose.observability.yml ps\n"
                "\n"
                "Check collector logs:\n"
                "  docker-compose -f docker-compose.observability.yml logs otel-collector"
            )
        except requests.exceptions.Timeout:
            result.errors.append("Collector health check timeout")
            result.recommendations.append(
                "Check if collector container is running:\n"
                "  docker ps | grep otel-collector"
            )
        except Exception as e:
            result.warnings.append(f"Health check error: {str(e)}")

    # 3. Check OTLP endpoints
    if check_connectivity:
        # Check OTLP HTTP endpoint
        http_open = check_port_open(collector_host, collector_http_port)
        if http_open:
            result.otlp_http_accessible = True
        else:
            result.errors.append(
                f"OTLP HTTP endpoint not accessible (port {collector_http_port})"
            )
            result.recommendations.append(
                f"Verify port {collector_http_port} is exposed in docker-compose.observability.yml\n"
                "Check port is not in use: lsof -i :4318"
            )

        # Check OTLP gRPC endpoint
        grpc_open = check_port_open(collector_host, collector_grpc_port)
        if grpc_open:
            result.otlp_grpc_accessible = True
        else:
            result.warnings.append(
                f"OTLP gRPC endpoint not accessible (port {collector_grpc_port})"
            )
            result.recommendations.append(
                f"Note: gRPC endpoint (port {collector_grpc_port}) is optional if using HTTP"
            )

    # 4. Check backend services
    if check_backends and HAS_REQUESTS:
        # Check Grafana
        try:
            response = requests.get(f"{final_grafana}/api/health", timeout=3)
            if response.status_code == 200:
                result.grafana_accessible = True
            else:
                result.warnings.append(f"Grafana returned HTTP {response.status_code}")
        except Exception:
            result.warnings.append("Grafana not accessible")
            result.recommendations.append(
                "Grafana should be available at http://localhost:3000\n"
                "Check container is running: docker ps | grep grafana"
            )

        # Check Tempo
        try:
            # Tempo doesn't have a dedicated health endpoint, check if port is open
            if check_port_open(collector_host, 3200):
                result.tempo_accessible = True
            else:
                result.warnings.append("Tempo not accessible (port 3200)")
        except Exception:
            result.warnings.append("Tempo connectivity check failed")

        # Check Loki
        try:
            if check_port_open(collector_host, 3100):
                result.loki_accessible = True
            else:
                result.warnings.append("Loki not accessible (port 3100)")
        except Exception:
            result.warnings.append("Loki connectivity check failed")

        # Check Mimir
        try:
            if check_port_open(collector_host, 9009):
                result.mimir_accessible = True
            else:
                result.warnings.append("Mimir not accessible (port 9009)")
        except Exception:
            result.warnings.append("Mimir connectivity check failed")

    # 5. Check OpenTelemetry dependencies
    try:
        import opentelemetry
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError:
        result.warnings.append("OpenTelemetry not installed")
        result.recommendations.append(
            "Install OpenTelemetry for full functionality:\n"
            "  pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp"
        )

    # 6. Additional recommendations
    if result.collector_healthy and result.otlp_http_accessible and not result.errors:
        result.recommendations.append(
            "✅ Setup validated successfully! Next steps:\n"
            "  • Run quickstart example: python examples/quickstarts/otel_collector_quickstart.py\n"
            "  • Open Grafana at http://localhost:3000 (admin/genops)\n"
            "  • Navigate to 'GenOps AI - Governance Overview' dashboard\n"
            "  • Explore traces in Tempo via Grafana → Explore → Tempo"
        )

    # Handle backend warnings
    if result.warnings and not result.errors:
        backend_warnings = [w for w in result.warnings if "not accessible" in w.lower()]
        if backend_warnings:
            result.recommendations.append(
                "Some backend services are not accessible.\n"
                "This is OK for basic testing, but for full observability:\n"
                "  • Start all services: docker-compose -f docker-compose.observability.yml up -d\n"
                "  • Verify all containers: docker-compose -f docker-compose.observability.yml ps"
            )

    # Final validation status
    if check_connectivity:
        # Full validation requires collector health and at least HTTP endpoint
        result.valid = (
            result.collector_healthy
            and result.otlp_http_accessible
            and not result.errors
        )
    else:
        # Config-only validation just checks for errors
        result.valid = not result.errors

    return result


def print_validation_result(result: OTelCollectorValidationResult) -> None:
    """
    Print validation result in user-friendly format.

    Args:
        result: Validation result to print

    Example:
        >>> result = validate_setup()
        >>> print_validation_result(result)

        OpenTelemetry Collector Validation Report
        ============================================================
        [SUCCESS] Collector Status: Healthy
        [SUCCESS] OTLP HTTP Endpoint: Accessible
        ...
    """
    print("\n" + "=" * 70)
    print("OpenTelemetry Collector Validation Report")
    print("=" * 70)
    print()

    # Collector status
    if result.collector_healthy:
        print("✅ [SUCCESS] Collector Status: Healthy")
        if result.collector_version:
            print(f"✅ [SUCCESS] Collector Version: {result.collector_version}")
    else:
        print("❌ [ERROR] Collector Status: Not Healthy")

    # OTLP endpoints
    if result.otlp_http_accessible:
        print("✅ [SUCCESS] OTLP HTTP Endpoint: Accessible (port 4318)")
    else:
        print("❌ [ERROR] OTLP HTTP Endpoint: Not Accessible")

    if result.otlp_grpc_accessible:
        print("✅ [SUCCESS] OTLP gRPC Endpoint: Accessible (port 4317)")
    elif result.otlp_http_accessible:
        print("ℹ️  [INFO] OTLP gRPC Endpoint: Not checked (HTTP is sufficient)")

    # Backend services
    if result.grafana_accessible:
        print("✅ [SUCCESS] Grafana: Accessible (http://localhost:3000)")
    elif result.warnings and any("Grafana" in w for w in result.warnings):
        print("⚠️  [WARNING] Grafana: Not Accessible")

    if result.tempo_accessible:
        print("✅ [SUCCESS] Tempo: Accessible (http://localhost:3200)")
    elif result.warnings and any("Tempo" in w for w in result.warnings):
        print("⚠️  [WARNING] Tempo: Not Accessible")

    if result.loki_accessible:
        print("✅ [SUCCESS] Loki: Accessible (http://localhost:3100)")

    if result.mimir_accessible:
        print("✅ [SUCCESS] Mimir: Accessible (http://localhost:9009)")

    print()

    # Errors
    if result.errors:
        print("❌ ERRORS:")
        print("-" * 70)
        for i, error in enumerate(result.errors, 1):
            print(f"{i}. {error}")
        print()

    # Warnings
    if result.warnings:
        print("⚠️  WARNINGS:")
        print("-" * 70)
        for i, warning in enumerate(result.warnings, 1):
            print(f"{i}. {warning}")
        print()

    # Recommendations
    if result.recommendations:
        print("💡 RECOMMENDATIONS:")
        print("-" * 70)
        for i, rec in enumerate(result.recommendations, 1):
            # Handle multi-line recommendations
            lines = rec.split("\n")
            for j, line in enumerate(lines):
                if j == 0:
                    print(f"{i}. {line}")
                else:
                    print(f"   {line}")
        print()

    # Overall status
    print("=" * 70)
    if result.valid:
        print("✅ [SUCCESS] Validation: PASSED")
        print("   Ready to send GenOps telemetry to OTel Collector!")
    else:
        print("❌ [ERROR] Validation: FAILED")
        print("   Fix the errors above before proceeding.")
    print("=" * 70)
    print()


def get_quickstart_instructions() -> str:
    """
    Get quickstart instructions for OTel Collector setup.

    Returns:
        Formatted instructions string
    """
    return """
=======================================================================
OpenTelemetry Collector Quickstart Instructions
=======================================================================

If validation failed, follow these steps:

1. Start the Observability Stack:
   ─────────────────────────────────────────────────────────────────
   cd /path/to/GenOps-AI-OTel
   docker-compose -f docker-compose.observability.yml up -d

2. Verify Containers Are Running:
   ─────────────────────────────────────────────────────────────────
   docker-compose -f docker-compose.observability.yml ps

   Expected: All services should show "Up" status

3. Check Service Logs (if issues):
   ─────────────────────────────────────────────────────────────────
   docker-compose -f docker-compose.observability.yml logs otel-collector
   docker-compose -f docker-compose.observability.yml logs grafana

4. Test Individual Services:
   ─────────────────────────────────────────────────────────────────
   # OTel Collector health
   curl http://localhost:13133/

   # Grafana health
   curl http://localhost:3000/api/health

   # OTLP HTTP endpoint
   curl -v http://localhost:4318/v1/traces

5. Run Validation Again:
   ─────────────────────────────────────────────────────────────────
   python examples/observability/validate_otel_collector.py

6. Run Quickstart Example:
   ─────────────────────────────────────────────────────────────────
   python examples/quickstarts/otel_collector_quickstart.py

7. Open Grafana Dashboard:
   ─────────────────────────────────────────────────────────────────
   Open: http://localhost:3000
   Login: admin / genops
   Dashboard: GenOps AI - Governance Overview

=======================================================================
Need Help?
=======================================================================

Documentation: docs/otel-collector-quickstart.md
GitHub Issues: https://github.com/KoshiHQ/GenOps-AI/issues
Discussions: https://github.com/KoshiHQ/GenOps-AI/discussions

=======================================================================
"""
