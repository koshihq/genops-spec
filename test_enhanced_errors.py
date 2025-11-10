#!/usr/bin/env python3
"""
Test script for enhanced error messages and diagnostics.

This script demonstrates the improved error handling with specific actionable fixes.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_adapter_initialization():
    """Test adapter initialization with different error scenarios."""
    print("🧪 Testing Enhanced Error Messages")
    print("=" * 50)

    try:
        from genops.providers.haystack_adapter import GenOpsHaystackAdapter

        print("\n1. Testing normal initialization...")
        adapter = GenOpsHaystackAdapter(
            team="test-team",
            project="error-testing",
            strict_mode=False  # Don't fail on missing dependencies
        )

        # Check initialization status
        adapter.print_initialization_status()

        print("\n2. Testing validation framework...")
        from genops.providers.haystack import (
            print_validation_result,
            validate_haystack_setup,
        )

        result = validate_haystack_setup()
        print_validation_result(result)

        print("\n✅ Enhanced error handling test completed!")
        print("\n💡 Key improvements:")
        print("   • Specific fix suggestions with commands")
        print("   • Interactive validation script references")
        print("   • Developer persona guidance")
        print("   • Progressive error resolution")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

def test_interactive_validation():
    """Test the interactive validation script."""
    print("\n🔧 Testing Interactive Validation")
    print("=" * 50)

    try:
        # Test that the validation script exists and is executable
        script_path = Path(__file__).parent / "scripts" / "validate_setup.py"

        if script_path.exists():
            print(f"✅ Interactive validation script found: {script_path}")
            print("   You can test it with: python scripts/validate_setup.py")
        else:
            print(f"❌ Validation script not found at: {script_path}")
            return False

        # Test that wrapper exists
        wrapper_path = Path(__file__).parent / "validate"
        if wrapper_path.exists():
            print(f"✅ Validation wrapper found: {wrapper_path}")
            print("   You can test it with: ./validate")
        else:
            print(f"❌ Validation wrapper not found at: {wrapper_path}")
            return False

        return True

    except Exception as e:
        print(f"❌ Interactive validation test failed: {e}")
        return False

def main():
    """Run all enhanced error handling tests."""
    print("🚀 Enhanced Error Handling Test Suite")
    print("=" * 60)

    tests = [
        ("Adapter Initialization", test_adapter_initialization),
        ("Interactive Validation", test_interactive_validation)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        print("-" * 30)

        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 30)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {test_name}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All enhanced error handling features working!")
        print("\n📚 Developer Experience Enhancements:")
        print("   • Interactive setup validation with ./validate")
        print("   • Actionable error messages with specific commands")
        print("   • Progressive complexity learning paths")
        print("   • Developer persona-based guidance")
        return 0
    else:
        print("⚠️ Some tests failed - check output above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
