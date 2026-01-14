"""
Cribl Stream Setup Validation

This module provides comprehensive validation for GenOps → Cribl Stream integration,
following the Universal Validation Framework from CLAUDE.md.
"""

import os
import socket
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class ValidationLevel(Enum):
    """Validation issue severity levels."""
    ERROR = "error"      # Blocks operation
    WARNING = "warning"  # Degraded functionality
    INFO = "info"        # Optimization suggestion


@dataclass
class ValidationIssue:
    """Single validation issue with fix suggestion."""
    level: ValidationLevel
    component: str
    message: str
    fix_suggestion: str


@dataclass
class ValidationResult:
    """Complete validation result with all issues."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    summary: str = ""

    def add_issue(self, level: ValidationLevel, component: str,
                  message: str, fix_suggestion: str):
        """Add a validation issue."""
        self.issues.append(ValidationIssue(level, component, message, fix_suggestion))
        if level == ValidationLevel.ERROR:
            self.is_valid = False


def validate_setup(
    endpoint: Optional[str] = None,
    auth_token: Optional[str] = None
) -> ValidationResult:
    """
    Validate GenOps → Cribl Stream setup.

    Checks:
    - Environment variables set correctly
    - Cribl endpoint reachable
    - OTLP port accessible
    - Authentication token valid format
    - Network connectivity

    Args:
        endpoint: Cribl OTLP endpoint (default: CRIBL_OTLP_ENDPOINT env var)
        auth_token: Bearer token (default: CRIBL_AUTH_TOKEN env var)

    Returns:
        ValidationResult with is_valid and list of issues
    """
    result = ValidationResult(is_valid=True)

    # Check 1: Environment variables
    endpoint = endpoint or os.getenv("CRIBL_OTLP_ENDPOINT")
    auth_token = auth_token or os.getenv("CRIBL_AUTH_TOKEN")

    if not endpoint:
        result.add_issue(
            ValidationLevel.ERROR,
            "Configuration",
            "CRIBL_OTLP_ENDPOINT not set",
            "Set environment variable: export CRIBL_OTLP_ENDPOINT='http://cribl-stream:4318'"
        )

    if not auth_token:
        result.add_issue(
            ValidationLevel.WARNING,
            "Authentication",
            "CRIBL_AUTH_TOKEN not set - using anonymous mode",
            "Set token for production: export CRIBL_AUTH_TOKEN='your-token'"
        )

    if not endpoint:
        result.summary = "Configuration incomplete"
        return result

    # Check 2: Parse endpoint and extract host/port
    try:
        from urllib.parse import urlparse
        parsed = urlparse(endpoint)
        host = parsed.hostname or "localhost"
        port = parsed.port or 4318
    except Exception as e:
        result.add_issue(
            ValidationLevel.ERROR,
            "Configuration",
            f"Invalid endpoint URL format: {endpoint}",
            "Use format: http://cribl-stream:4318 or https://cribl-cloud.example.com:4318"
        )
        result.summary = "Invalid endpoint configuration"
        return result

    # Check 3: Network connectivity (DNS resolution)
    try:
        socket.gethostbyname(host)
    except socket.gaierror:
        result.add_issue(
            ValidationLevel.ERROR,
            "Connectivity",
            f"Cannot resolve hostname: {host}",
            f"Verify DNS: ping {host} or check /etc/hosts"
        )
        result.summary = "Cannot reach Cribl endpoint"
        return result

    # Check 4: Port accessibility (TCP connect)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        sock.connect((host, port))
        sock.close()
    except (socket.timeout, ConnectionRefusedError, OSError) as e:
        result.add_issue(
            ValidationLevel.ERROR,
            "Connectivity",
            f"Cannot connect to {host}:{port}",
            f"Check Cribl Stream is running and port {port} is open. Test with: telnet {host} {port}"
        )
        result.summary = "Cribl endpoint not reachable"
        return result

    # Check 5: Auth token format (if provided)
    if auth_token:
        if len(auth_token) < 16:
            result.add_issue(
                ValidationLevel.WARNING,
                "Authentication",
                "Auth token seems too short (< 16 characters)",
                "Verify token from Cribl UI: Settings → Authentication"
            )

        # Check for common mistakes
        if auth_token.startswith("Bearer "):
            result.add_issue(
                ValidationLevel.WARNING,
                "Authentication",
                "Token includes 'Bearer ' prefix - this will be added automatically",
                "Remove 'Bearer ' prefix from token: export CRIBL_AUTH_TOKEN='token-only'"
            )

    # Check 6: GenOps dependencies
    try:
        import opentelemetry
        from opentelemetry.sdk.trace import TracerProvider
    except ImportError:
        result.add_issue(
            ValidationLevel.ERROR,
            "Dependencies",
            "OpenTelemetry SDK not installed",
            "Install: pip install opentelemetry-api opentelemetry-sdk"
        )

    # Final summary
    error_count = sum(1 for issue in result.issues if issue.level == ValidationLevel.ERROR)
    warning_count = sum(1 for issue in result.issues if issue.level == ValidationLevel.WARNING)

    if result.is_valid:
        result.summary = "✅ All checks passed"
        if warning_count > 0:
            result.summary += f" ({warning_count} warnings)"
    else:
        result.summary = f"❌ {error_count} errors, {warning_count} warnings"

    return result


def print_validation_result(result: ValidationResult) -> None:
    """
    Print validation result in user-friendly format.

    Shows:
    - Success/failure status with color
    - Each issue with severity indicator
    - Specific fix suggestion for each issue
    - Links to documentation
    """
    print("=" * 70)
    print("CRIBL STREAM SETUP VALIDATION")
    print("=" * 70)
    print()

    # Overall status
    if result.is_valid:
        print("✅ Status: PASSED")
    else:
        print("❌ Status: FAILED")

    print(f"Summary: {result.summary}")
    print()

    if not result.issues:
        print("No issues found - you're ready to send telemetry!")
        print()
        print("Next steps:")
        print("  1. Run the quickstart example: python examples/observability/cribl_integration.py")
        print("  2. Check Cribl UI for incoming events: Data → Sources → genops_otlp_source → Live Data")
        print("  3. Configure pipelines: docs/integrations/cribl.md")
        return

    # Group issues by level
    errors = [i for i in result.issues if i.level == ValidationLevel.ERROR]
    warnings = [i for i in result.issues if i.level == ValidationLevel.WARNING]
    infos = [i for i in result.issues if i.level == ValidationLevel.INFO]

    # Print errors
    if errors:
        print("🚨 ERRORS (must fix to proceed):")
        print()
        for i, issue in enumerate(errors, 1):
            print(f"{i}. [{issue.component}] {issue.message}")
            print(f"   Fix: {issue.fix_suggestion}")
            print()

    # Print warnings
    if warnings:
        print("⚠️  WARNINGS (recommended fixes):")
        print()
        for i, issue in enumerate(warnings, 1):
            print(f"{i}. [{issue.component}] {issue.message}")
            print(f"   Fix: {issue.fix_suggestion}")
            print()

    # Print infos
    if infos:
        print("💡 SUGGESTIONS:")
        print()
        for i, issue in enumerate(infos, 1):
            print(f"{i}. [{issue.component}] {issue.message}")
            print(f"   Tip: {issue.fix_suggestion}")
            print()

    # Documentation links
    print("=" * 70)
    print("📚 Documentation:")
    print("  - Quickstart: docs/cribl-quickstart.md")
    print("  - Full guide: docs/integrations/cribl.md")
    print("  - Troubleshooting: docs/integrations/cribl.md#troubleshooting")
    print("=" * 70)
