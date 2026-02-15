"""
Direct OTLP export to Grafana Tempo example.

This example demonstrates:
- Direct trace export to Tempo (bypassing OTel Collector)
- Validation of Tempo connectivity
- Basic span creation with governance attributes

Prerequisites:
    - Tempo running at http://localhost:3200
    - OTLP receiver enabled on port 4318

Quick start Tempo:
    docker run -d -p 3200:3200 -p 4318:4318 grafana/tempo:latest
"""

import time

from genops.integrations.tempo import (
    configure_tempo,
    print_tempo_validation,
    validate_tempo_setup,
)

from genops import track_usage


def main():
    """
    Direct export example with comprehensive setup validation.
    """
    print("=" * 60)
    print("Grafana Tempo Direct Export Example")
    print("=" * 60)
    print()

    # Step 1: Validate Tempo is accessible
    print("Step 1: Validating Tempo setup...")
    print("-" * 60)

    result = validate_tempo_setup(tempo_endpoint="http://localhost:3200")
    print_tempo_validation(result)

    if not result.valid:
        print("❌ Tempo validation failed. Please fix issues above.")
        return

    # Step 2: Configure direct export to Tempo
    print("\nStep 2: Configuring direct export to Tempo...")
    print("-" * 60)

    configure_tempo(
        endpoint="http://localhost:3200",
        service_name="tempo-direct-export-example",
        environment="development",
    )

    print("✅ Configured direct OTLP export to Tempo")
    print()

    # Step 3: Create sample spans with GenOps tracking
    print("\nStep 3: Creating sample traces...")
    print("-" * 60)

    @track_usage(
        team="platform-engineering",
        project="tempo-examples",
        customer_id="internal-testing",
        feature="direct-export",
    )
    def example_ai_operation():
        """Simulated AI operation with governance tracking."""
        print("  → Executing example AI operation...")
        time.sleep(0.1)  # Simulate work
        return {"status": "success", "tokens": 1500}

    # Execute tracked operation
    result = example_ai_operation()
    print(f"  ✅ Operation completed: {result}")

    # Give time for span export
    print("\n  ⏳ Waiting for span export to Tempo...")
    time.sleep(2)

    # Step 4: Verify traces in Tempo
    print("\nStep 4: Verify traces in Tempo...")
    print("-" * 60)
    print("Query traces using:")
    print("  1. TraceQL (command line):")
    print(
        '     curl "http://localhost:3200/api/search?q={.team=\\"platform-engineering\\"}&limit=10"'
    )
    print()
    print("  2. Grafana UI:")
    print("     http://localhost:3000 → Explore → Tempo")
    print()

    print("=" * 60)
    print("✅ Direct export example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
