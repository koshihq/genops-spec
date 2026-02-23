"""Validation utilities for Collibra integration setup."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import urlparse

from genops.providers.collibra.client import CollibraAPIClient, CollibraAPIError


@dataclass
class CollibraValidationResult:
    """Result of Collibra setup validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    connectivity: bool = False
    api_version: str | None = None
    available_domains: list[str] = field(default_factory=list)
    policy_count: int = 0

    @property
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return len(self.warnings) > 0


def validate_url_format(url: str) -> tuple[bool, str | None]:
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
            return (
                False,
                f"Invalid URL scheme: {parsed.scheme} (expected http or https)",
            )
        if not parsed.netloc:
            return False, "URL missing domain"
        return True, None
    except Exception as e:
        return False, f"Invalid URL format: {str(e)}"


def validate_setup(
    collibra_url: str | None = None,
    username: str | None = None,
    password: str | None = None,
    api_token: str | None = None,
    check_connectivity: bool = True,
    check_permissions: bool = True,
) -> CollibraValidationResult:
    """
    Validate Collibra integration setup.

    Args:
        collibra_url: Collibra instance URL (or from COLLIBRA_URL env var)
        username: Username (or from COLLIBRA_USERNAME env var)
        password: Password (or from COLLIBRA_PASSWORD env var)
        api_token: API token (or from COLLIBRA_API_TOKEN env var)
        check_connectivity: Test API connectivity
        check_permissions: Check required permissions

    Returns:
        CollibraValidationResult with validation details
    """
    result = CollibraValidationResult(valid=False)

    # 1. Check environment variables
    env_url = os.getenv("COLLIBRA_URL")
    env_username = os.getenv("COLLIBRA_USERNAME")
    env_password = os.getenv("COLLIBRA_PASSWORD")
    env_token = os.getenv("COLLIBRA_API_TOKEN")

    # Use provided values or fall back to environment
    final_url = collibra_url or env_url
    final_username = username or env_username
    final_password = password or env_password
    final_token = api_token or env_token

    # Validate URL
    if not final_url:
        result.errors.append("COLLIBRA_URL not set")
        result.recommendations.append(
            'Set environment variable: export COLLIBRA_URL="https://your-instance.collibra.com"'
        )
    else:
        url_valid, url_error = validate_url_format(final_url)
        if not url_valid:
            result.errors.append(f"Invalid URL format: {url_error}")
        elif not final_url.startswith("https://"):
            result.warnings.append(
                "Using HTTP instead of HTTPS. Consider using HTTPS for security."
            )

    # Validate authentication
    has_basic_auth = final_username and final_password
    has_token_auth = final_token is not None

    if not has_basic_auth and not has_token_auth:
        result.errors.append("No authentication credentials provided")
        result.recommendations.append(
            "Set credentials:\n"
            "  export COLLIBRA_USERNAME='your-username'\n"
            "  export COLLIBRA_PASSWORD='your-password'\n"
            "Or use API token:\n"
            "  export COLLIBRA_API_TOKEN='your-api-token'"
        )
    elif has_basic_auth and has_token_auth:
        result.warnings.append(
            "Both basic auth and token provided. Token will be used."
        )

    # If basic validation failed, return early
    if result.errors:
        return result

    # 2. Test connectivity
    if check_connectivity and final_url:
        try:
            client = CollibraAPIClient(
                base_url=final_url,
                username=final_username,
                password=final_password,
                api_token=final_token,
            )

            result.connectivity = client.health_check()

            if result.connectivity:
                # Get API version
                try:
                    app_info = client.get_application_info()
                    result.api_version = app_info.get("version", "unknown")
                except Exception:
                    result.warnings.append("Could not retrieve API version")

                # List available domains
                try:
                    domains = client.list_domains()
                    result.available_domains = [
                        f"{d.get('name', 'Unknown')} (id: {d.get('id', 'N/A')})"
                        for d in domains[:5]  # Limit to first 5
                    ]

                    if not domains:
                        result.warnings.append(
                            "No domains found. Create a domain in Collibra UI for AI governance."
                        )
                    elif len(domains) > 5:
                        result.available_domains.append(
                            f"... and {len(domains) - 5} more domains"
                        )
                except Exception as e:
                    result.warnings.append(f"Could not list domains: {str(e)}")

                # Check policy access
                if check_permissions:
                    try:
                        policies = client.list_policies()
                        result.policy_count = len(policies)

                        if result.policy_count == 0:
                            result.recommendations.append(
                                "No policies found. Create governance policies in Collibra to enable policy sync."
                            )
                    except Exception as e:
                        result.warnings.append(
                            f"Could not access policies: {str(e)}. "
                            "Policy import may not be available."
                        )

            else:
                result.errors.append("API health check failed")
                result.recommendations.append(
                    "Check Collibra URL and network connectivity:\n"
                    f"  URL: {final_url}\n"
                    "  Verify URL is correct and accessible"
                )

        except CollibraAPIError as e:
            result.connectivity = False
            result.errors.append(f"API connection failed: {e.message}")

            if e.status_code == 401:
                result.recommendations.append(
                    "Authentication failed. Check credentials:\n"
                    "  1. Verify username/password or API token\n"
                    "  2. Check if account has access to Collibra\n"
                    "  3. Verify credentials haven't expired"
                )
            elif e.status_code == 404:
                result.recommendations.append(
                    "API endpoint not found. Check Collibra URL:\n"
                    f"  Current: {final_url}\n"
                    "  Expected format: https://your-instance.collibra.com"
                )
            else:
                result.recommendations.append(
                    f"Connection error (status {e.status_code}). "
                    "Check network connectivity and firewall rules."
                )

        except Exception as e:
            result.connectivity = False
            result.errors.append(f"Unexpected error: {str(e)}")

    # 3. Additional recommendations
    if result.connectivity:
        if not result.warnings:
            result.recommendations.append(
                "Setup looks good! Consider:\n"
                "  • Enable batch export to reduce API calls\n"
                "  • Configure webhook endpoint for real-time policy updates\n"
                "  • Set up dedicated Collibra domain for AI governance"
            )

    # Final validation status
    result.valid = result.connectivity and not result.errors

    return result


def print_validation_result(result: CollibraValidationResult) -> None:
    """
    Print validation result in user-friendly format.

    Args:
        result: Validation result to print
    """
    print("\nCollibra Integration Validation Report")
    print("=" * 60)
    print()

    # Connection status
    if result.connectivity:
        print("[SUCCESS] Connection Status: Connected")
    else:
        print("[ERROR] Connection Status: Not Connected")

    # API version
    if result.api_version:
        print(f"[SUCCESS] API Version: {result.api_version}")

    # Available domains
    if result.available_domains:
        print(
            f"[SUCCESS] Available Domains: {len(result.available_domains)} domains accessible"
        )
        for domain in result.available_domains:
            print(f"   - {domain}")

    # Policy access
    if result.policy_count > 0:
        print(f"[SUCCESS] Policy Access: {result.policy_count} policies available")
    elif result.connectivity:
        print("[WARNING] Policy Access: No policies found")

    print()

    # Errors
    if result.errors:
        print("[ERROR] Errors:")
        for error in result.errors:
            print(f"  - {error}")
        print()

    # Warnings
    if result.warnings:
        print("[WARNING] Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
        print()

    # Recommendations
    if result.recommendations:
        print("[INFO] Recommendations:")
        for rec in result.recommendations:
            # Handle multi-line recommendations
            lines = rec.split("\n")
            for i, line in enumerate(lines):
                if i == 0:
                    print(f"  - {line}")
                else:
                    print(f"    {line}")
        print()

    # Overall status
    print("=" * 60)
    if result.valid:
        print("[SUCCESS] Validation: PASSED")
        print("   Ready to integrate GenOps with Collibra!")
    else:
        print("[ERROR] Validation: FAILED")
        print("   Fix the errors above before proceeding.")
    print("=" * 60)
    print()


def get_validation_script() -> str:
    """
    Get standalone validation script that can be run independently.

    Returns:
        Python script as string
    """
    return '''#!/usr/bin/env python3
"""
Collibra Integration Validation Script

Run this script to validate your Collibra setup:
    python validate_collibra_setup.py

Or with custom credentials:
    python validate_collibra_setup.py --url https://company.collibra.com --username user --password pass
"""

import sys
import argparse

try:
    from genops.providers.collibra.validation import validate_setup, print_validation_result
except ImportError:
    print("❌ GenOps not installed. Install with: pip install genops[collibra]")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Validate Collibra integration setup")
    parser.add_argument("--url", help="Collibra instance URL")
    parser.add_argument("--username", help="Collibra username")
    parser.add_argument("--password", help="Collibra password")
    parser.add_argument("--api-token", help="Collibra API token")
    parser.add_argument("--no-connectivity", action="store_true", help="Skip connectivity check")

    args = parser.parse_args()

    result = validate_setup(
        collibra_url=args.url,
        username=args.username,
        password=args.password,
        api_token=args.api_token,
        check_connectivity=not args.no_connectivity
    )

    print_validation_result(result)

    sys.exit(0 if result.valid else 1)


if __name__ == "__main__":
    main()
'''
