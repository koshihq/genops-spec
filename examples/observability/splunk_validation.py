"""Validation utilities for Splunk HEC integration setup."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urlparse

# SSL warnings will be shown when verify_ssl=False to ensure users are aware of security implications

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class SplunkValidationResult:
    """Result of Splunk HEC setup validation."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    connectivity: bool = False
    hec_version: Optional[str] = None
    index_accessible: bool = False

    @property
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return len(self.warnings) > 0


def validate_url_format(url: str) -> tuple[bool, Optional[str]]:
    """
    Validate Splunk HEC endpoint URL format.

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
    splunk_hec_endpoint: Optional[str] = None,
    splunk_hec_token: Optional[str] = None,
    splunk_index: str = "genops_ai",
    check_connectivity: bool = True,
    verify_ssl: bool = True,
) -> SplunkValidationResult:
    """
    Validate Splunk HEC integration setup.

    This function performs comprehensive validation of your Splunk HEC configuration:
    1. Environment variables (SPLUNK_HEC_ENDPOINT, SPLUNK_HEC_TOKEN)
    2. URL format validation
    3. HEC health check (/services/collector/health)
    4. Token authentication test
    5. Index write permissions
    6. OpenTelemetry dependencies

    Args:
        splunk_hec_endpoint: Splunk HEC endpoint URL (or from SPLUNK_HEC_ENDPOINT env var)
        splunk_hec_token: HEC authentication token (or from SPLUNK_HEC_TOKEN env var)
        splunk_index: Target Splunk index for telemetry data
        check_connectivity: Test API connectivity and authentication
        verify_ssl: Verify SSL certificates (default: True). Set to False only for
                   self-signed certificates in trusted environments. This is a security risk.

    Returns:
        SplunkValidationResult with validation details

    Example:
        >>> result = validate_setup()
        >>> if result.valid:
        ...     print("Setup validated successfully!")
        >>> else:
        ...     for error in result.errors:
        ...         print(f"Error: {error}")

        >>> # For self-signed certificates (development/trusted environments only)
        >>> result = validate_setup(verify_ssl=False)
    """
    result = SplunkValidationResult(valid=False)

    # Security warning for disabled SSL verification
    if not verify_ssl:
        result.warnings.append(
            "⚠️  SSL certificate verification is DISABLED. "
            "This is insecure and should only be used in trusted environments "
            "with self-signed certificates."
        )

    # Check if requests library is available
    if check_connectivity and not HAS_REQUESTS:
        result.errors.append("requests library not installed")
        result.recommendations.append(
            "Install requests: pip install requests\n"
            "Or skip connectivity check: validate_setup(check_connectivity=False)"
        )
        return result

    # 1. Check environment variables
    env_endpoint = os.getenv("SPLUNK_HEC_ENDPOINT")
    env_token = os.getenv("SPLUNK_HEC_TOKEN")

    # Use provided values or fall back to environment
    final_endpoint = splunk_hec_endpoint or env_endpoint
    final_token = splunk_hec_token or env_token

    # Validate endpoint
    if not final_endpoint:
        result.errors.append("SPLUNK_HEC_ENDPOINT not set")
        result.recommendations.append(
            'Set environment variable:\n'
            '  export SPLUNK_HEC_ENDPOINT="https://splunk.example.com:8088"'
        )
    else:
        url_valid, url_error = validate_url_format(final_endpoint)
        if not url_valid:
            result.errors.append(f"Invalid endpoint URL: {url_error}")
            result.recommendations.append(
                f"Current endpoint: {final_endpoint}\n"
                "Expected format: https://splunk.example.com:8088"
            )
        elif not final_endpoint.startswith("https://"):
            result.warnings.append(
                "Using HTTP instead of HTTPS. Consider using HTTPS for security."
            )

    # Validate token
    if not final_token:
        result.errors.append("SPLUNK_HEC_TOKEN not set")
        result.recommendations.append(
            'Set environment variable:\n'
            '  export SPLUNK_HEC_TOKEN="your-hec-token"\n'
            '\n'
            'To create HEC token in Splunk:\n'
            '  1. Navigate to Settings → Data Inputs → HTTP Event Collector\n'
            '  2. Click "New Token"\n'
            '  3. Configure and save token'
        )

    # If basic validation failed, return early
    if result.errors:
        return result

    # 2. Test connectivity
    if check_connectivity and HAS_REQUESTS:
        try:
            # HEC health check
            health_url = f"{final_endpoint}/services/collector/health"

            try:
                response = requests.get(health_url, verify=verify_ssl, timeout=5)
            except requests.exceptions.SSLError as ssl_error:
                # SSL verification failed - provide helpful error
                result.errors.append(
                    "SSL certificate verification failed"
                )
                result.recommendations.append(
                    "SSL certificate verification failed:\n"
                    f"  Error: {str(ssl_error)}\n"
                    "  Solutions:\n"
                    "    1. Use valid SSL certificate (recommended)\n"
                    "    2. For self-signed certificates in trusted environments:\n"
                    "       validate_setup(verify_ssl=False)\n"
                    "    3. Set REQUESTS_CA_BUNDLE environment variable to CA certificate path"
                )
                return result

            if response.status_code == 200:
                result.connectivity = True
                try:
                    health_data = response.json()
                    result.hec_version = health_data.get("text", "HEC is healthy")
                except Exception:
                    result.hec_version = "HEC is healthy"
            else:
                result.errors.append(
                    f"HEC health check failed (HTTP {response.status_code})"
                )
                result.recommendations.append(
                    f"Health check URL: {health_url}\n"
                    f"Response: {response.text[:200] if response.text else 'No response body'}"
                )

        except requests.exceptions.Timeout:
            result.errors.append("Connection timeout - HEC endpoint not reachable")
            result.recommendations.append(
                "Troubleshooting steps:\n"
                "  • Check network connectivity\n"
                "  • Verify firewall rules allow outbound connections\n"
                "  • Confirm Splunk is running and accessible\n"
                f"  • Test manually: curl -k {final_endpoint}/services/collector/health"
            )
        except requests.exceptions.ConnectionError as e:
            result.errors.append("Connection refused - HEC endpoint not accessible")
            result.recommendations.append(
                f"Verify HEC endpoint configuration:\n"
                f"  Current endpoint: {final_endpoint}\n"
                "  Troubleshooting:\n"
                "    • Check Splunk is running\n"
                "    • Verify port 8088 is accessible\n"
                "    • Check firewall rules\n"
                "    • Confirm HEC is enabled in Splunk:\n"
                "      Settings → Data Inputs → HTTP Event Collector → Global Settings"
            )
        except Exception as e:
            result.errors.append(f"Unexpected connection error: {str(e)}")
            result.recommendations.append(
                "Check network configuration and Splunk availability"
            )

        # 3. Test token authentication
        if result.connectivity:
            try:
                test_url = f"{final_endpoint}/services/collector"
                headers = {"Authorization": f"Splunk {final_token}"}
                test_event = {
                    "event": "genops_validation_test",
                    "sourcetype": "_json",
                    "index": splunk_index
                }

                try:
                    response = requests.post(
                        test_url,
                        json=test_event,
                        headers=headers,
                        verify=verify_ssl,
                        timeout=5
                    )
                except requests.exceptions.SSLError as ssl_error:
                    result.errors.append(
                        "SSL certificate verification failed during token authentication"
                    )
                    result.recommendations.append(
                        f"SSL verification failed: {str(ssl_error)}\n"
                        "For self-signed certificates, use: validate_setup(verify_ssl=False)"
                    )
                    return result

                if response.status_code == 200:
                    result.index_accessible = True
                    response_data = response.json()
                    if response_data.get("code") == 0:
                        # Successful event ingestion
                        pass
                    else:
                        result.warnings.append(
                            f"Token test succeeded but returned code: {response_data.get('code')}"
                        )
                elif response.status_code == 401:
                    result.errors.append(
                        "HEC token authentication failed (401 Unauthorized)"
                    )
                    result.recommendations.append(
                        "Check HEC token configuration:\n"
                        "  1. In Splunk: Settings → Data Inputs → HTTP Event Collector\n"
                        "  2. Verify token exists and is enabled\n"
                        "  3. Check token hasn't expired\n"
                        "  4. Ensure Global Settings has HEC enabled"
                    )
                elif response.status_code == 403:
                    result.errors.append(
                        "HEC token forbidden (403 Forbidden)"
                    )
                    result.recommendations.append(
                        "Token exists but lacks permissions:\n"
                        f"  • Verify token has write permission to index '{splunk_index}'\n"
                        "  • Check token source type restrictions\n"
                        "  • Confirm index exists and is writable"
                    )
                elif response.status_code == 404:
                    result.errors.append(
                        "HEC endpoint not found (404 Not Found)"
                    )
                    result.recommendations.append(
                        f"Check endpoint URL: {test_url}\n"
                        "Verify HEC is enabled in Splunk Global Settings"
                    )
                else:
                    result.warnings.append(
                        f"Token test returned unexpected status: {response.status_code}"
                    )
                    result.recommendations.append(
                        f"Response: {response.text[:200] if response.text else 'No response body'}"
                    )

            except Exception as e:
                result.warnings.append(f"Token validation test failed: {str(e)}")
                result.recommendations.append(
                    "Unable to validate token authentication. "
                    "Manual verification recommended."
                )

    # 4. Check OpenTelemetry dependencies
    try:
        import opentelemetry
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError:
        result.warnings.append("OpenTelemetry not installed")
        result.recommendations.append(
            "Install OpenTelemetry for full functionality:\n"
            "  pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp"
        )

    # 5. Additional recommendations
    if result.connectivity and result.index_accessible and not result.errors:
        result.recommendations.append(
            "✅ Setup validated successfully! Next steps:\n"
            "  • Create dedicated index 'genops_ai' for better organization\n"
            "  • Configure index retention policies for compliance\n"
            "  • Set up alerting for budget thresholds\n"
            "  • Consider using Cribl for multi-destination routing\n"
            "  • Import dashboard templates from splunk_integration.py"
        )

    # Final validation status
    if check_connectivity:
        # Full validation requires connectivity and authentication
        result.valid = result.connectivity and result.index_accessible and not result.errors
    else:
        # Config-only validation just checks for errors
        result.valid = not result.errors

    return result


def print_validation_result(result: SplunkValidationResult) -> None:
    """
    Print validation result in user-friendly format.

    Args:
        result: Validation result to print

    Example:
        >>> result = validate_setup()
        >>> print_validation_result(result)

        Splunk HEC Integration Validation Report
        ============================================================
        [SUCCESS] HEC Status: Connected
        [SUCCESS] Index Access: Token authenticated successfully
        ...
    """
    print("\n" + "=" * 70)
    print("Splunk HEC Integration Validation Report")
    print("=" * 70)
    print()

    # Connection status
    if result.connectivity:
        print("✅ [SUCCESS] HEC Status: Connected")
        if result.hec_version:
            print(f"✅ [SUCCESS] HEC Version: {result.hec_version}")
    else:
        print("❌ [ERROR] HEC Status: Not Connected")

    # Index accessibility
    if result.index_accessible:
        print("✅ [SUCCESS] Index Access: Token authenticated successfully")
    elif result.connectivity:
        print("❌ [ERROR] Index Access: Token authentication failed")

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
        print("   Ready to send GenOps telemetry to Splunk!")
    else:
        print("❌ [ERROR] Validation: FAILED")
        print("   Fix the errors above before proceeding.")
    print("=" * 70)
    print()


def get_validation_script() -> str:
    """
    Get standalone validation script that can be run independently.

    Returns:
        Python script as string

    Example:
        Save this script and run it:
        >>> script = get_validation_script()
        >>> with open('validate_splunk.py', 'w') as f:
        ...     f.write(script)
        >>> # Then run: python validate_splunk.py
    """
    return '''#!/usr/bin/env python3
"""
Splunk HEC Integration Validation Script

Run this script to validate your Splunk HEC setup:
    python validate_splunk_setup.py

Or with custom credentials:
    python validate_splunk_setup.py --endpoint https://splunk.example.com:8088 --token YOUR_TOKEN
"""

import sys
import argparse

try:
    from splunk_validation import validate_setup, print_validation_result
except ImportError:
    print("❌ Validation module not found.")
    print("   Ensure splunk_validation.py is in the same directory.")
    print("   Or install: pip install genops-ai")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Validate Splunk HEC integration setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate using environment variables
  python validate_splunk_setup.py

  # Validate with explicit credentials
  python validate_splunk_setup.py \\
    --endpoint https://splunk.example.com:8088 \\
    --token YOUR_HEC_TOKEN \\
    --index genops_ai

  # Skip connectivity check (validate config only)
  python validate_splunk_setup.py --no-connectivity
        """
    )
    parser.add_argument(
        "--endpoint",
        help="Splunk HEC endpoint URL (e.g., https://splunk.example.com:8088)"
    )
    parser.add_argument(
        "--token",
        help="Splunk HEC authentication token"
    )
    parser.add_argument(
        "--index",
        default="genops_ai",
        help="Target Splunk index (default: genops_ai)"
    )
    parser.add_argument(
        "--no-connectivity",
        action="store_true",
        help="Skip connectivity and authentication checks"
    )

    args = parser.parse_args()

    print("🔍 Validating Splunk HEC integration setup...")
    print()

    result = validate_setup(
        splunk_hec_endpoint=args.endpoint,
        splunk_hec_token=args.token,
        splunk_index=args.index,
        check_connectivity=not args.no_connectivity
    )

    print_validation_result(result)

    sys.exit(0 if result.valid else 1)


if __name__ == "__main__":
    main()
'''
