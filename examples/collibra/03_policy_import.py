#!/usr/bin/env python3
"""
Collibra Policy Import Example

This example demonstrates importing governance policies FROM Collibra
and enforcing them on AI operations using GenOps PolicyEngine.

Prerequisites:
    export COLLIBRA_URL="https://your-instance.collibra.com"
    export COLLIBRA_USERNAME="your-username"
    export COLLIBRA_PASSWORD="your-password"
    # OR use API token:
    export COLLIBRA_API_TOKEN="your-api-token"

    export GENOPS_TEAM="your-team"
    export GENOPS_PROJECT="your-project"

Run this example:
    python 03_policy_import.py
"""

import os
import sys
import time

from genops.core.policy import PolicyViolationError, check_policy


def print_header() -> None:
    """Print example header."""
    print("=" * 70)
    print("Collibra Policy Import + Enforcement")
    print("=" * 70)
    print()


def check_prerequisites() -> bool:
    """Check if required dependencies are available."""
    print("Prerequisites Check:")
    print()

    missing = []

    # Check GenOps
    try:
        import genops  # noqa: F401

        print("  [OK] GenOps installed")
    except ImportError:
        print("  [ERROR] GenOps not installed")
        missing.append("pip install genops")

    # Check Collibra credentials
    has_basic_auth = os.getenv("COLLIBRA_USERNAME") and os.getenv("COLLIBRA_PASSWORD")
    has_token = os.getenv("COLLIBRA_API_TOKEN")

    if os.getenv("COLLIBRA_URL"):
        print("  [OK] COLLIBRA_URL configured")
    else:
        print("  [ERROR] COLLIBRA_URL not set")
        missing.append('export COLLIBRA_URL="https://your-instance.collibra.com"')

    if has_basic_auth or has_token:
        auth_method = "token" if has_token else "basic auth"
        print(f"  [OK] Collibra credentials configured ({auth_method})")
    else:
        print("  [ERROR] Collibra credentials not configured")
        missing.append('export COLLIBRA_USERNAME="your-username"')
        missing.append('export COLLIBRA_PASSWORD="your-password"')

    if missing:
        print("\n[ERROR] Missing requirements:")
        for req in missing:
            print(f"   {req}")
        return False

    print("  [OK] All prerequisites met!")
    print()
    return True


def demonstrate_policy_import() -> int:
    """Demonstrate policy import and enforcement."""
    print("Policy Import + Enforcement Demo")
    print()

    try:
        from genops.providers.collibra import GenOpsCollibraAdapter

        # ============================================================
        # STEP 1: Enable policy sync during adapter initialization
        # ============================================================
        print("Step 1: Initialize adapter with policy sync enabled")
        print("-" * 70)

        adapter = GenOpsCollibraAdapter(
            team=os.getenv("GENOPS_TEAM", "ml-platform"),
            project=os.getenv("GENOPS_PROJECT", "ai-governance-demo"),
            enable_policy_sync=True,  # Enable policy import
            policy_sync_interval_minutes=5,  # Sync every 5 minutes
            export_mode="batch",
        )

        print("[OK] Adapter initialized with policy sync enabled")
        print("   • Policies imported from Collibra on startup")
        print("   • Background sync every 5 minutes")
        print("   • Policies automatically registered with GenOps PolicyEngine")
        print()

        # ============================================================
        # STEP 2: View imported policies
        # ============================================================
        print("Step 2: View policies imported from Collibra")
        print("-" * 70)

        if adapter.policy_importer:
            imported_policies = adapter.policy_importer.get_imported_policies()
            stats = adapter.policy_importer.get_stats()

            print(f"   Policies Imported: {stats.policies_imported}")
            print(f"   Import Failures: {stats.policies_failed}")
            print()

            if imported_policies:
                print("   Imported Policy Details:")
                for policy_name, policy_config in imported_policies.items():
                    print(f"      • {policy_name}")
                    print(f"        - Description: {policy_config.description}")
                    print(f"        - Enabled: {policy_config.enabled}")
                    print(
                        f"        - Enforcement: {policy_config.enforcement_level.value}"
                    )
                    print(f"        - Conditions: {policy_config.conditions}")
                    print()
            else:
                print("   [INFO]  No policies imported yet")
                print(
                    "      (Ensure policies exist in your Collibra domain with recognized types)"
                )
                print()
        else:
            print("   [ERROR] Policy importer not initialized")
            print()

        # ============================================================
        # STEP 3: Enforce policies on AI operations
        # ============================================================
        print("Step 3: Enforce policies on AI operations")
        print("-" * 70)

        # Example 1: Check cost limit policy
        print("   Example 1: Cost limit policy enforcement")
        policy_result = check_policy(
            "cost_limit",
            {"cost": 5.0},  # Estimated operation cost
        )
        print("      Policy: cost_limit")
        print(f"      Result: {policy_result.result.value}")
        if policy_result.reason:
            print(f"      Reason: {policy_result.reason}")
        print()

        # Example 2: Check rate limit policy
        print("   Example 2: Rate limit policy enforcement")
        policy_result = check_policy(
            "rate_limit",
            {
                "request_count": 50,  # Current requests
            },
        )
        print("      Policy: rate_limit")
        print(f"      Result: {policy_result.result.value}")
        if policy_result.reason:
            print(f"      Reason: {policy_result.reason}")
        print()

        # Example 3: Check team access policy
        print("   Example 3: Team access policy enforcement")
        policy_result = check_policy(
            "team_access",
            {
                "team": os.getenv("GENOPS_TEAM", "ml-platform"),
            },
        )
        print("      Policy: team_access")
        print(f"      Result: {policy_result.result.value}")
        if policy_result.reason:
            print(f"      Reason: {policy_result.reason}")
        print()

        # ============================================================
        # STEP 4: AI operations with automatic policy enforcement
        # ============================================================
        print("Step 4: AI operations with policy enforcement")
        print("-" * 70)

        print("   Operation 1: Low-cost completion (should pass)")
        try:
            with adapter.track_ai_operation("low-cost-completion") as span:
                time.sleep(0.1)

                # Record cost
                adapter.record_cost(span, cost=0.02, provider="openai", model="gpt-4")

                # Record policy evaluation (simulated)
                adapter.record_policy(
                    span,
                    policy_name="cost_limit",
                    policy_result="allowed",
                    policy_reason="Within cost limit",
                )

            print("      [OK] Operation completed successfully")
        except PolicyViolationError as e:
            print(f"      [ERROR] Policy violation: {e}")
        print()

        print("   Operation 2: High-cost operation (may be blocked)")
        try:
            with adapter.track_ai_operation("high-cost-operation") as span:
                time.sleep(0.1)

                # Record cost
                adapter.record_cost(
                    span, cost=50.0, provider="anthropic", model="claude-3-opus"
                )

                # Record policy evaluation (simulated)
                adapter.record_policy(
                    span,
                    policy_name="cost_limit",
                    policy_result="warning",
                    policy_reason="High cost operation",
                )

            print("      [OK] Operation completed with warning")
        except PolicyViolationError as e:
            print(f"      [ERROR] Policy violation: {e}")
        print()

        # ============================================================
        # STEP 5: Manual policy sync
        # ============================================================
        print("Step 5: Manual policy synchronization")
        print("-" * 70)

        sync_result = adapter.sync_policies()

        print("   Sync Results:")
        print(f"      • Policies Imported: {sync_result.get('imported', 0)}")
        print(f"      • Policies Updated: {sync_result.get('updated', 0)}")
        print(f"      • Failures: {sync_result.get('failed', 0)}")
        print()

        # ============================================================
        # STEP 6: View policy sync statistics
        # ============================================================
        print("Step 6: Policy sync statistics")
        print("-" * 70)

        if adapter.policy_importer:
            stats = adapter.policy_importer.get_stats()

            print(f"   Total Policies Imported: {stats.policies_imported}")
            print(f"   Total Policies Updated: {stats.policies_updated}")
            print(f"   Total Failures: {stats.policies_failed}")
            if stats.last_sync_time:
                time_since_sync = time.time() - stats.last_sync_time
                print(f"   [TIME] Last Sync: {time_since_sync:.1f} seconds ago")

            if stats.errors:
                print("   [ERROR] Recent Errors:")
                for error in stats.errors[-5:]:  # Show last 5 errors
                    print(f"      • {error}")
        print()

        # Cleanup
        adapter.shutdown()
        print("[OK] Adapter shutdown complete")
        print()

        return True

    except ImportError as e:
        print(f"[ERROR] Required package not available: {e}")
        print("   Fix: pip install genops")
        return False
    except Exception as e:
        print(f"[ERROR] Error during policy import demo: {e}")
        import traceback

        traceback.print_exc()
        return False


def print_next_steps() -> None:
    """Print next steps."""
    print()
    print("=" * 70)
    print("Policy Import Demo Complete!")
    print("=" * 70)
    print()
    print("What just happened?")
    print("  1. [OK] Adapter imported policies from Collibra on initialization")
    print("  2. [OK] Policies registered with GenOps PolicyEngine for enforcement")
    print("  3. [OK] Policies checked against AI operations before execution")
    print("  4. [OK] Background sync keeps policies up-to-date automatically")
    print()
    print("Key Capabilities:")
    print("  • Bidirectional Integration: Export telemetry + Import policies")
    print("  • Automatic Policy Sync: Collibra policies → GenOps enforcement")
    print("  • Runtime Enforcement: Policies applied to AI operations in real-time")
    print("  • Periodic Updates: Background sync keeps policies current")
    print("  • Manual Sync: On-demand policy refresh when needed")
    print()
    print("Policy Types Supported:")
    print("  • AI Cost Limit: Maximum cost per operation")
    print("  • AI Rate Limit: Request rate throttling")
    print("  • Content Filter: Blocked content patterns")
    print("  • Team Access Control: Team-based authorization")
    print("  • Budget Constraint: Daily/monthly budget limits")
    print("  • Model Governance: Allowed/blocked model restrictions")
    print()
    print("Next Steps:")
    print("  1. Create governance policies in Collibra UI")
    print("  2. Policies automatically sync to GenOps")
    print("  3. Policies enforce at runtime on AI operations")
    print("  4. Telemetry exports back to Collibra for audit")
    print("  5. Explore 04_bidirectional_sync.py for full workflow")
    print()
    print("Documentation:")
    print("  • Policy Configuration: docs/policies/collibra-policy-mapping.md")
    print("  • Full Integration Guide: docs/integrations/collibra.md")
    print()


def main() -> int:
    """Main policy import workflow."""
    print_header()

    # Check prerequisites
    if not check_prerequisites():
        print("[ERROR] Prerequisites not met.")
        print()
        print("Quick Fix:")
        print("  1. Set COLLIBRA_URL, COLLIBRA_USERNAME, COLLIBRA_PASSWORD")
        print("  2. Run again: python 03_policy_import.py")
        return 1

    # Run policy import demo
    success = demonstrate_policy_import()

    if success:
        print_next_steps()
        return 0
    else:
        print("=" * 70)
        print("[ERROR] Policy import demo failed!")
        print("=" * 70)
        print()
        print("Troubleshooting:")
        print("  1. Verify Collibra credentials are correct")
        print("  2. Check that policies exist in Collibra domain")
        print("  3. Ensure policy types match recognized types:")
        print("     - AI Cost Limit")
        print("     - AI Rate Limit")
        print("     - Content Filter")
        print("     - Team Access Control")
        print("     - Budget Constraint")
        print("     - Model Governance")
        print("  4. Run validation:")
        print("     python -m genops.providers.collibra.validation")
        print()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
