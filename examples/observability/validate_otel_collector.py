#!/usr/bin/env python3
"""
OpenTelemetry Collector Integration Validation Script

Run this script to validate your OTel Collector setup:
    python validate_otel_collector.py

Or with custom endpoint:
    python validate_otel_collector.py --endpoint http://localhost:4318

For configuration-only validation (skip connectivity):
    python validate_otel_collector.py --no-connectivity
"""

import argparse
import sys

try:
    from otel_collector_validation import (
        get_quickstart_instructions,
        print_validation_result,
        validate_setup,
    )
except ImportError:
    print("❌ Validation module not found.")
    print("   Ensure otel_collector_validation.py is in the same directory.")
    print("   Or install: pip install genops-ai")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Validate OpenTelemetry Collector integration setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate using default local setup
  python validate_otel_collector.py

  # Validate with custom collector endpoint
  python validate_otel_collector.py --endpoint http://collector.example.com:4318

  # Validate with custom Grafana endpoint
  python validate_otel_collector.py --grafana http://grafana.example.com:3000

  # Skip connectivity check (validate config only)
  python validate_otel_collector.py --no-connectivity

  # Skip backend service checks
  python validate_otel_collector.py --no-backends

  # Verbose output with quickstart instructions
  python validate_otel_collector.py --verbose
        """,
    )
    parser.add_argument(
        "--endpoint", help="OTel Collector OTLP endpoint (e.g., http://localhost:4318)"
    )
    parser.add_argument(
        "--grafana", help="Grafana endpoint (default: http://localhost:3000)"
    )
    parser.add_argument(
        "--no-connectivity",
        action="store_true",
        help="Skip connectivity and health checks",
    )
    parser.add_argument(
        "--no-backends",
        action="store_true",
        help="Skip backend service checks (Grafana, Tempo, Loki, Mimir)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show quickstart instructions after validation",
    )

    args = parser.parse_args()

    print("🔍 Validating OpenTelemetry Collector integration setup...")
    print()

    result = validate_setup(
        collector_endpoint=args.endpoint,
        grafana_endpoint=args.grafana,
        check_connectivity=not args.no_connectivity,
        check_backends=not args.no_backends,
    )

    print_validation_result(result)

    # Show quickstart instructions if validation failed and verbose mode
    if not result.valid or args.verbose:
        print(get_quickstart_instructions())

    # Exit with appropriate code
    sys.exit(0 if result.valid else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        print(
            "   Please report this issue: https://github.com/KoshiHQ/GenOps-AI/issues"
        )
        sys.exit(1)
