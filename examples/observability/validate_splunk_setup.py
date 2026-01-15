#!/usr/bin/env python3
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
    parser.add_argument(
        "--no-ssl-verify",
        action="store_true",
        help="Disable SSL certificate verification (insecure, use only with self-signed certificates)"
    )

    args = parser.parse_args()

    print("🔍 Validating Splunk HEC integration setup...")
    print()

    result = validate_setup(
        splunk_hec_endpoint=args.endpoint,
        splunk_hec_token=args.token,
        splunk_index=args.index,
        check_connectivity=not args.no_connectivity,
        verify_ssl=not args.no_ssl_verify
    )

    print_validation_result(result)

    sys.exit(0 if result.valid else 1)


if __name__ == "__main__":
    main()
