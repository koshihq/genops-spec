#!/usr/bin/env python3
"""Quick test to verify core functionality works."""

import os
import sys

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_basic_imports():
    """Test that basic imports work."""
    print("🧪 Testing basic imports...")

    try:
        import genops

        print(f"✅ genops imported - version {genops.__version__}")

        # Test core functions are available
        assert hasattr(genops, "track_usage")
        assert hasattr(genops, "track")
        assert hasattr(genops, "enforce_policy")
        assert hasattr(genops, "init")
        assert hasattr(genops, "status")
        print("✅ All core functions available")

        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("\n🧪 Testing basic functionality...")

    try:
        import genops

        # Test status function
        status = genops.status()
        assert isinstance(status, dict)
        assert "initialized" in status
        print("✅ Status function works")

        # Test telemetry creation
        from genops.core.telemetry import GenOpsTelemetry

        telemetry = GenOpsTelemetry()
        assert telemetry is not None
        print("✅ Telemetry engine works")

        # Test policy configuration
        from genops.core.policy import PolicyConfig, PolicyResult

        policy = PolicyConfig(
            name="test_policy", enforcement_level=PolicyResult.BLOCKED
        )
        assert policy.name == "test_policy"
        print("✅ Policy engine works")

        return True
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_provider_adapters():
    """Test provider adapters handle missing dependencies gracefully."""
    print("\n🧪 Testing provider adapters...")

    try:
        # Test OpenAI adapter import (should work even without openai package)
        from genops.providers.openai import GenOpsOpenAIAdapter

        print("✅ OpenAI adapter imports")

        # Test Anthropic adapter import
        from genops.providers.anthropic import GenOpsAnthropicAdapter

        print("✅ Anthropic adapter imports")

        # Creating adapters without dependencies should fail gracefully
        try:
            GenOpsOpenAIAdapter()
            print("⚠️ OpenAI adapter created (openai package must be installed)")
        except ImportError:
            print("✅ OpenAI adapter properly handles missing dependency")

        try:
            GenOpsAnthropicAdapter()
            print("⚠️ Anthropic adapter created (anthropic package must be installed)")
        except ImportError:
            print("✅ Anthropic adapter properly handles missing dependency")

        return True
    except Exception as e:
        print(f"❌ Provider adapter test failed: {e}")
        return False


def main():
    """Run quick validation tests."""
    print("🚀 GenOps AI Quick Validation Tests")
    print("=" * 40)

    success = True

    success &= test_basic_imports()
    success &= test_basic_functionality()
    success &= test_provider_adapters()

    print("\n" + "=" * 40)

    if success:
        print("🎉 All quick tests passed!")
        print("✅ GenOps AI is ready for comprehensive testing")
    else:
        print("❌ Some quick tests failed")
        print("🔧 Please fix issues before running full test suite")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
