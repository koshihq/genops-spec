#!/usr/bin/env python3
"""Test script for GenOps auto-instrumentation."""

import os
import sys

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_auto_instrumentation():
    """Test the auto-instrumentation system."""
    print("🧪 Testing GenOps Auto-Instrumentation System")
    print("=" * 50)

    # Test import
    import genops

    print("✅ GenOps imported successfully")

    # Test status before initialization
    status = genops.status()
    print(f"Status before init: initialized={status['initialized']}")
    print(f"Available providers: {status.get('available_providers', {})}")

    # Test initialization
    print("\n📦 Initializing GenOps...")
    genops.init(
        service_name="test-service",
        environment="testing",
        default_team="test-team",
        default_project="auto-init-test",
        exporter_type="console",
    )
    print("✅ genops.init() completed")

    # Test status after initialization
    status = genops.status()
    print("\nStatus after init:")
    print(f"  Initialized: {status['initialized']}")
    print(f"  Available providers: {status['available_providers']}")
    print(f"  Instrumented providers: {status['instrumented_providers']}")

    # Test default attributes
    defaults = genops.get_default_attributes()
    print(f"  Default attributes: {defaults}")

    # Test manual instrumentation with defaults
    print("\n🔧 Testing manual instrumentation with defaults...")

    @genops.track_usage(operation_name="test_operation", feature="auto-init-testing")
    def test_function():
        return "Test completed successfully"

    result = test_function()
    print(f"✅ Manual instrumentation result: {result}")

    # Test uninstrumentation
    print("\n🔄 Testing uninstrumentation...")
    genops.uninstrument()

    status = genops.status()
    print(f"Status after uninstrument: initialized={status['initialized']}")

    print("\n🎉 All auto-instrumentation tests passed!")
    return True


if __name__ == "__main__":
    try:
        success = test_auto_instrumentation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
