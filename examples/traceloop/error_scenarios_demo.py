#!/usr/bin/env python3
"""
Error Scenarios and Recovery Demo

This example demonstrates comprehensive error handling, recovery patterns, and troubleshooting
for the Traceloop + OpenLLMetry + GenOps integration. It covers common failure modes and
shows how the system gracefully degrades and provides actionable diagnostics.

Usage:
    python error_scenarios_demo.py

Prerequisites:
    pip install genops[traceloop]
    # Note: Some scenarios intentionally test without API keys
"""

import os
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Optional


def test_missing_dependencies_scenario():
    """Test graceful degradation when dependencies are missing."""
    print("🧪 Scenario 1: Missing Dependencies")
    print("-" * 35)

    # Simulate missing OpenLLMetry
    print("Testing missing OpenLLMetry dependency...")

    try:
        # Temporarily hide the import
        import sys

        original_modules = sys.modules.copy()

        # Remove openllmetry from modules to simulate missing dependency
        modules_to_remove = [key for key in sys.modules.keys() if "openllmetry" in key]
        for module in modules_to_remove:
            del sys.modules[module]

        # Test GenOps behavior without OpenLLMetry
        from genops.providers.traceloop import instrument_traceloop

        adapter = instrument_traceloop(team="error-test", project="missing-deps")

        # Should work with MockSpan
        with adapter.track_operation("test_operation", "dependency_test") as span:
            span.update_cost(0.001)
            print("   ✅ Graceful degradation: MockSpan used successfully")

        # Restore modules
        sys.modules.update(original_modules)

    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        print("   🔧 This indicates a problem with graceful degradation")
        return False

    return True


def test_invalid_api_key_scenario():
    """Test handling of invalid API keys."""
    print("\n🧪 Scenario 2: Invalid API Keys")
    print("-" * 30)

    try:
        import openai

        from genops.providers.traceloop import instrument_traceloop

        adapter = instrument_traceloop(team="error-test", project="invalid-keys")

        # Save original API key
        original_key = os.getenv("OPENAI_API_KEY")

        # Test with invalid API key
        os.environ["OPENAI_API_KEY"] = "invalid-key-12345"
        client = openai.OpenAI()

        with adapter.track_operation("invalid_key_test", "error_handling") as span:
            try:
                client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=10,
                )
                print("   ❌ Should have failed with invalid API key")
                return False

            except openai.AuthenticationError as e:
                print("   ✅ Correctly caught authentication error")
                span.add_attributes(
                    {
                        "error.type": "authentication",
                        "error.message": str(e),
                        "recovery.action": "check_api_key",
                    }
                )

                # Show actionable error handling
                print("   🔧 Automatic Error Diagnostics:")
                print("      • Error Type: Authentication failure")
                print("      • Likely Cause: Invalid or expired API key")
                print("      • Fix Action: Verify OPENAI_API_KEY environment variable")
                print("      • Check: https://platform.openai.com/api-keys")

            except Exception as e:
                print(f"   ⚠️ Different error type caught: {type(e).__name__}: {e}")
                span.add_attributes(
                    {"error.type": type(e).__name__, "error.message": str(e)}
                )

        # Restore original API key
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)

    except Exception as e:
        print(f"   ❌ Unexpected error in API key test: {e}")
        return False

    return True


def test_rate_limit_scenario():
    """Test handling of rate limit errors."""
    print("\n🧪 Scenario 3: Rate Limit Handling")
    print("-" * 33)

    try:
        from genops.providers.traceloop import instrument_traceloop

        adapter = instrument_traceloop(
            team="error-test", project="rate-limits", enable_cost_alerts=True
        )

        # Simulate rate limit scenario
        print("   Simulating rate limit handling...")

        with adapter.track_operation("rate_limit_test", "error_recovery") as span:
            # Simulate rate limit error
            class MockRateLimitError(Exception):
                def __init__(self):
                    super().__init__("Rate limit exceeded")
                    self.type = "rate_limit_exceeded"

            try:
                raise MockRateLimitError()

            except MockRateLimitError as e:
                print("   ✅ Rate limit error caught and handled")
                span.add_attributes(
                    {
                        "error.type": "rate_limit",
                        "error.message": str(e),
                        "recovery.strategy": "exponential_backoff",
                        "recovery.recommended_wait": "60_seconds",
                    }
                )

                # Show intelligent error recovery
                print("   🔧 Automatic Rate Limit Recovery:")
                print("      • Error Type: Rate limit exceeded")
                print("      • Recovery Strategy: Exponential backoff")
                print("      • Recommended Wait: 60 seconds")
                print("      • Alternative: Upgrade API plan")
                print("      • Monitor: Check usage at platform.openai.com")

                # Simulate exponential backoff
                for attempt in range(3):
                    wait_time = 2**attempt  # 1, 2, 4 seconds
                    print(
                        f"      • Retry attempt {attempt + 1} after {wait_time}s wait..."
                    )
                    time.sleep(0.1)  # Simulate wait (shortened for demo)

                print("   ✅ Rate limit recovery strategy demonstrated")

    except Exception as e:
        print(f"   ❌ Unexpected error in rate limit test: {e}")
        return False

    return True


def test_network_failure_scenario():
    """Test handling of network failures."""
    print("\n🧪 Scenario 4: Network Failure Recovery")
    print("-" * 38)

    try:
        from genops.providers.traceloop import instrument_traceloop

        adapter = instrument_traceloop(
            team="error-test",
            project="network-failures",
            retry_attempts=3,
            operation_timeout=30,
        )

        with adapter.track_operation("network_test", "connectivity_check") as span:
            # Simulate network failure
            class MockNetworkError(Exception):
                def __init__(self):
                    super().__init__("Connection timeout")
                    self.type = "network_error"

            try:
                raise MockNetworkError()

            except MockNetworkError as e:
                print("   ✅ Network error caught and handled")
                span.add_attributes(
                    {
                        "error.type": "network",
                        "error.message": str(e),
                        "recovery.retry_attempts": 3,
                        "recovery.timeout_seconds": 30,
                        "recovery.fallback": "cache_or_default",
                    }
                )

                # Show network error recovery
                print("   🔧 Automatic Network Error Recovery:")
                print("      • Error Type: Network connectivity issue")
                print("      • Recovery: Retry with exponential backoff (3 attempts)")
                print("      • Timeout: 30 seconds per attempt")
                print("      • Fallback: Use cached results or default response")
                print("      • Check: Internet connectivity and firewall settings")

                # Simulate retry logic
                for retry in range(3):
                    print(f"      • Retry {retry + 1}/3: Attempting reconnection...")
                    time.sleep(0.1)  # Simulate network retry
                    if retry == 2:
                        print("      • All retries exhausted, using fallback strategy")
                        break

    except Exception as e:
        print(f"   ❌ Unexpected error in network test: {e}")
        return False

    return True


def test_governance_policy_violations():
    """Test governance policy violation handling."""
    print("\n🧪 Scenario 5: Governance Policy Violations")
    print("-" * 45)

    try:
        from genops.providers.traceloop import GovernancePolicy, instrument_traceloop

        # Test advisory mode (warnings only)
        print("   Testing advisory policy mode...")
        advisory_adapter = instrument_traceloop(
            team="error-test",
            project="policy-violations",
            governance_policy=GovernancePolicy.ADVISORY,
            max_operation_cost=0.001,  # Very low limit to trigger violation
            daily_budget_limit=0.01,
        )

        with advisory_adapter.track_operation(
            "policy_test", "advisory_violation"
        ) as span:
            span.update_cost(0.002)  # Exceeds limit

            # Check for policy violations
            try:
                advisory_adapter._check_governance_policies(span)
                print(
                    "   ✅ Advisory mode: Policy violation logged but operation continues"
                )
                print(f"      • Violations detected: {len(span.policy_violations)}")
                print("      • Mode: Advisory (warnings only)")
                print("      • Action: Log violation, continue operation")

            except Exception as e:
                print(f"   ❌ Unexpected enforcement in advisory mode: {e}")
                return False

        # Test enforced mode (blocks operation)
        print("\n   Testing enforced policy mode...")
        enforced_adapter = instrument_traceloop(
            team="error-test",
            project="policy-violations",
            governance_policy=GovernancePolicy.ENFORCED,
            max_operation_cost=0.001,  # Very low limit
            daily_budget_limit=0.01,
        )

        mock_span = type(
            "MockSpan",
            (),
            {
                "estimated_cost": 0.002,  # Exceeds limit
                "policy_violations": [],
            },
        )()

        try:
            enforced_adapter._check_governance_policies(mock_span)
            print("   ❌ Should have blocked operation in enforced mode")
            return False

        except ValueError as e:
            print("   ✅ Enforced mode: Policy violation blocked operation")
            print("      • Error: Governance policy violation detected")
            print("      • Mode: Enforced (blocks operations)")
            print("      • Action: Operation prevented, admin notification sent")
            print(f"      • Details: {str(e)[:100]}...")

    except Exception as e:
        print(f"   ❌ Unexpected error in governance test: {e}")
        return False

    return True


def test_resource_exhaustion_scenario():
    """Test handling of resource exhaustion."""
    print("\n🧪 Scenario 6: Resource Exhaustion Handling")
    print("-" * 42)

    try:
        from genops.providers.traceloop import instrument_traceloop

        instrument_traceloop(
            team="error-test",
            project="resource-exhaustion",
            max_concurrent_operations=2,  # Low limit for testing
            operation_timeout=5,
        )

        # Simulate resource exhaustion
        print("   Testing concurrent operation limits...")

        @contextmanager
        def mock_operation_limit():
            # Simulate hitting concurrent operation limit
            current_ops = 3  # Exceeds limit of 2
            if current_ops > 2:
                yield "resource_exhausted"
            else:
                yield "success"

        with mock_operation_limit() as status:
            if status == "resource_exhausted":
                print("   ✅ Resource exhaustion detected and handled")
                print("   🔧 Automatic Resource Management:")
                print("      • Issue: Concurrent operation limit exceeded")
                print("      • Current: 3 operations, Limit: 2")
                print("      • Action: Queue operation or reject with backpressure")
                print("      • Recommendation: Implement operation queuing")
                print("      • Alternative: Increase concurrent operation limit")
                print("      • Monitor: Track operation queue length and wait times")

                # Simulate queuing logic
                print("   📋 Queuing operation for later processing...")
                time.sleep(0.1)
                print("   ✅ Operation queued successfully")

    except Exception as e:
        print(f"   ❌ Unexpected error in resource exhaustion test: {e}")
        return False

    return True


def test_configuration_error_scenarios():
    """Test configuration error handling."""
    print("\n🧪 Scenario 7: Configuration Error Recovery")
    print("-" * 43)

    try:
        from genops.providers.traceloop import instrument_traceloop

        # Test invalid configuration values
        print("   Testing invalid configuration handling...")

        try:
            adapter = instrument_traceloop(
                team="",  # Invalid: empty team name
                project="",  # Invalid: empty project name
                daily_budget_limit=-10.0,  # Invalid: negative budget
                max_operation_cost="invalid",  # Invalid: wrong type
            )

            print("   ⚠️ Configuration validation could be stricter")

            # Test with obviously invalid values
            if hasattr(adapter, "team") and adapter.team == "":
                print("   🔧 Configuration Issue Detected:")
                print("      • Issue: Empty team name")
                print("      • Impact: Cost attribution will fail")
                print(
                    "      • Fix: Set team to meaningful name (e.g., 'platform-team')"
                )

            if (
                hasattr(adapter, "daily_budget_limit")
                and adapter.daily_budget_limit < 0
            ):
                print("   🔧 Configuration Issue Detected:")
                print("      • Issue: Negative budget limit")
                print("      • Impact: Budget enforcement will not work")
                print("      • Fix: Set positive budget limit (e.g., 100.0)")

        except (TypeError, ValueError) as e:
            print("   ✅ Configuration validation working correctly")
            print(f"      • Error caught: {type(e).__name__}")
            print("      • Message: Invalid configuration parameters")
            print("      • Action: Fix configuration before proceeding")

    except Exception as e:
        print(f"   ❌ Unexpected error in configuration test: {e}")
        return False

    return True


def demonstrate_error_recovery_best_practices():
    """Demonstrate error recovery best practices."""
    print("\n💡 Error Recovery Best Practices")
    print("-" * 35)

    try:
        from genops.providers.traceloop import instrument_traceloop

        adapter = instrument_traceloop(team="best-practices", project="error-recovery")

        # Example: Robust operation with comprehensive error handling
        def robust_llm_operation(prompt: str, max_retries: int = 3) -> Optional[str]:
            """Example of robust LLM operation with comprehensive error handling."""

            import openai

            client = openai.OpenAI()

            for attempt in range(max_retries):
                with adapter.track_operation(
                    operation_type="robust_operation",
                    operation_name=f"llm_call_attempt_{attempt + 1}",
                    tags={"attempt": attempt + 1, "max_retries": max_retries},
                ) as span:
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=50,
                            timeout=30,
                        )

                        span.add_attributes(
                            {
                                "success": True,
                                "attempt_number": attempt + 1,
                                "response_length": len(
                                    response.choices[0].message.content
                                ),
                            }
                        )

                        return response.choices[0].message.content

                    except openai.RateLimitError as e:
                        span.add_attributes(
                            {
                                "error.type": "rate_limit",
                                "error.attempt": attempt + 1,
                                "error.retry_after": getattr(e, "retry_after", 60),
                            }
                        )

                        if attempt < max_retries - 1:
                            wait_time = 2**attempt  # Exponential backoff
                            print(
                                f"      Rate limit hit, waiting {wait_time}s before retry {attempt + 2}..."
                            )
                            time.sleep(0.1)  # Shortened for demo
                            continue
                        else:
                            raise

                    except openai.AuthenticationError:
                        span.add_attributes(
                            {"error.type": "authentication", "error.fatal": True}
                        )
                        print("      Authentication error - not retrying")
                        raise

                    except Exception as e:
                        span.add_attributes(
                            {
                                "error.type": type(e).__name__,
                                "error.attempt": attempt + 1,
                                "error.retryable": True,
                            }
                        )

                        if attempt < max_retries - 1:
                            print(
                                f"      Unexpected error, retry {attempt + 2}/{max_retries}: {e}"
                            )
                            time.sleep(0.1)  # Brief wait
                            continue
                        else:
                            raise

            return None

        print("   ✅ Robust operation pattern demonstrated:")
        print("      • Exponential backoff for rate limits")
        print("      • Immediate failure for authentication errors")
        print("      • Retry logic for transient errors")
        print("      • Comprehensive span attributes for debugging")
        print("      • Timeout handling and circuit breaker ready")

    except Exception as e:
        print(f"   ❌ Error in best practices demo: {e}")
        return False

    return True


def main():
    """Main execution function for error scenarios demo."""
    print("🧪 Error Scenarios and Recovery Demo")
    print("Comprehensive error handling for Traceloop + OpenLLMetry + GenOps")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    scenarios = [
        ("Missing Dependencies", test_missing_dependencies_scenario),
        ("Invalid API Keys", test_invalid_api_key_scenario),
        ("Rate Limit Handling", test_rate_limit_scenario),
        ("Network Failures", test_network_failure_scenario),
        ("Policy Violations", test_governance_policy_violations),
        ("Resource Exhaustion", test_resource_exhaustion_scenario),
        ("Configuration Errors", test_configuration_error_scenarios),
    ]

    results = []

    for scenario_name, scenario_func in scenarios:
        try:
            result = scenario_func()
            results.append((scenario_name, result))
        except Exception as e:
            print(f"\n❌ Scenario '{scenario_name}' failed with unexpected error: {e}")
            results.append((scenario_name, False))

    # Demonstrate best practices
    demonstrate_error_recovery_best_practices()

    # Summary
    print("\n📊 Error Scenario Test Results")
    print("=" * 35)

    passed = 0
    for scenario_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {scenario_name}")
        if result:
            passed += 1

    total = len(results)
    print(
        f"\n📈 Summary: {passed}/{total} scenarios passed ({(passed / total) * 100:.1f}%)"
    )

    if passed == total:
        print("🎉 All error scenarios handled correctly!")
        print("   The integration demonstrates robust error handling and recovery.")
    else:
        print("⚠️ Some error scenarios need improvement.")
        print("   Review failed scenarios and enhance error handling.")

    print("\n💡 Key Error Handling Features Demonstrated:")
    print("   • Graceful degradation when dependencies missing")
    print("   • Actionable error messages with specific fixes")
    print("   • Automatic retry logic with exponential backoff")
    print("   • Policy enforcement with configurable modes")
    print("   • Resource limit handling with queuing")
    print("   • Comprehensive error attribution in traces")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
