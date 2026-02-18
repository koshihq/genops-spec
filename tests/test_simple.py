#!/usr/bin/env python3
"""Simple working test to demonstrate test framework functionality."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_genops_basic_functionality():
    """Test basic GenOps functionality that we know works."""

    # Test basic imports
    import genops
    from genops.core.policy import PolicyConfig, PolicyEngine, PolicyResult
    from genops.core.telemetry import GenOpsTelemetry

    # Test telemetry creation
    telemetry = GenOpsTelemetry()
    assert telemetry is not None

    # Test policy creation
    policy = PolicyConfig(
        name="test_policy",
        description="A test policy",
        enforcement_level=PolicyResult.BLOCKED,
    )
    assert policy.name == "test_policy"
    assert policy.enforcement_level == PolicyResult.BLOCKED

    # Test policy engine
    engine = PolicyEngine()
    engine.register_policy(policy)
    assert "test_policy" in engine.policies

    # Test policy evaluation returns PolicyEvaluationResult
    eval_result = engine.evaluate_policy("test_policy", {})
    assert isinstance(eval_result.result, PolicyResult)

    # Test auto-instrumentation status
    status_info = genops.status()
    assert isinstance(status_info, dict)
    assert "initialized" in status_info

    print("✅ All basic functionality tests passed!")
    return True


if __name__ == "__main__":
    try:
        success = test_genops_basic_functionality()
        print("🎉 GenOps AI test framework is working correctly!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
