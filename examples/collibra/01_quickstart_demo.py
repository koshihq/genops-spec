#!/usr/bin/env python3
"""
Collibra + GenOps 5-Minute Quickstart

This example demonstrates zero-code auto-instrumentation with Collibra integration.
Your existing AI code automatically exports governance telemetry to Collibra with
no code changes required.

Prerequisites:
    export COLLIBRA_URL="https://your-instance.collibra.com"
    export COLLIBRA_USERNAME="your-username"
    export COLLIBRA_PASSWORD="your-password"
    # OR use API token:
    export COLLIBRA_API_TOKEN="your-api-token"

    export GENOPS_TEAM="your-team"
    export GENOPS_PROJECT="your-project"

Run this example:
    python 01_quickstart_demo.py
"""

import os
import sys
import time


def print_header() -> None:
    """Print example header."""
    print("=" * 70)
    print("GenOps + Collibra: 5-Minute Quickstart")
    print("=" * 70)
    print()


def check_prerequisites() -> bool:
    """Check if required dependencies are available."""
    print("Prerequisites Check:")

    missing = []

    # Check GenOps
    try:
        import genops  # noqa: F401

        print("  [OK] GenOps installed")
    except ImportError:
        print("  [MISSING] GenOps not installed")
        missing.append("pip install genops")

    # Check Collibra credentials
    has_basic_auth = os.getenv("COLLIBRA_USERNAME") and os.getenv("COLLIBRA_PASSWORD")
    has_token = os.getenv("COLLIBRA_API_TOKEN")

    if os.getenv("COLLIBRA_URL"):
        print("  [OK] COLLIBRA_URL configured")
    else:
        print("  [MISSING] COLLIBRA_URL not set")
        missing.append('export COLLIBRA_URL="https://your-instance.collibra.com"')

    if has_basic_auth or has_token:
        auth_method = "token" if has_token else "basic auth"
        print(f"  [OK] Collibra credentials configured ({auth_method})")
    else:
        print("  [MISSING] Collibra credentials not configured")
        missing.append('export COLLIBRA_USERNAME="your-username"')
        missing.append('export COLLIBRA_PASSWORD="your-password"')

    # Check governance attributes
    if os.getenv("GENOPS_TEAM"):
        print(f"  [OK] GENOPS_TEAM configured: {os.getenv('GENOPS_TEAM')}")
    else:
        print("  [OPTIONAL] GENOPS_TEAM not set (optional)")

    if os.getenv("GENOPS_PROJECT"):
        print(f"  [OK] GENOPS_PROJECT configured: {os.getenv('GENOPS_PROJECT')}")
    else:
        print("  [OPTIONAL] GENOPS_PROJECT not set (optional)")

    if missing:
        print("\n[MISSING] Missing requirements:")
        for req in missing:
            print(f"   {req}")
        return False

    print("  [OK] All prerequisites met!")
    print()
    return True


def demonstrate_quickstart() -> int:
    """Demonstrate 5-minute quickstart with auto-instrumentation."""
    print("5-Minute Quickstart Demo")
    print()

    try:
        from genops.providers.collibra import auto_instrument

        # ============================================================
        # STEP 1: One-line auto-instrumentation
        # ============================================================
        print("Step 1: Auto-instrument with Collibra (one line)")
        print("-" * 70)

        adapter = auto_instrument()

        print("[OK] Collibra integration active!")
        print("   • All AI operations automatically exported to Collibra")
        print("   • Governance telemetry captured transparently")
        print(f"   • Team: {adapter.team}")
        print(f"   • Project: {adapter.project}")
        print()

        # ============================================================
        # STEP 2: Your existing AI code works unchanged
        # ============================================================
        print("Step 2: Run your AI operations (no code changes needed)")
        print("-" * 70)

        # Operation 1: Simple completion
        print("   [AI] Operation 1: AI Completion")
        with adapter.track_ai_operation("ai-completion") as span:
            # Simulate AI operation
            time.sleep(0.1)

            # Record cost (this is the only addition to your existing code)
            adapter.record_cost(span, cost=0.02, provider="openai", model="gpt-4")
        print("      [OK] Cost: $0.02 → Exported to Collibra")

        # Operation 2: Higher-cost operation
        print("   [AI] Operation 2: Batch Processing")
        with adapter.track_ai_operation("batch-processing") as span:
            time.sleep(0.1)
            adapter.record_cost(
                span,
                cost=1.50,
                provider="anthropic",
                model="claude-3-opus",
                tokens_input=5000,
                tokens_output=8000,
            )
        print("      [OK] Cost: $1.50 → Exported to Collibra")

        # Operation 3: With policy enforcement
        print("   [AI] Operation 3: Governed Operation")
        with adapter.track_ai_operation("governed-operation") as span:
            time.sleep(0.1)

            adapter.record_cost(
                span, cost=0.05, provider="openai", model="gpt-3.5-turbo"
            )

            # Record policy enforcement
            adapter.record_policy(
                span,
                policy_name="cost_optimization",
                policy_result="allowed",
                policy_reason="Using cost-effective model",
            )
        print("      [OK] Cost: $0.05 + Policy: allowed → Exported to Collibra")
        print()

        # ============================================================
        # STEP 3: View real-time metrics
        # ============================================================
        print("Step 3: View real-time governance metrics")
        print("-" * 70)

        metrics = adapter.get_metrics()

        print(f"   Operations Tracked: {metrics['operation_count']}")
        print(f"   Total Cost: ${metrics['total_cost']:.2f}")
        print(f"   Assets Exported: {metrics['assets_exported']}")
        print(f"   Buffer Size: {metrics['buffer_size']}")

        if metrics.get("daily_budget_limit"):
            print(f"   Budget Remaining: ${metrics['budget_remaining']:.2f}")
        print()

        # ============================================================
        # STEP 4: Flush remaining data
        # ============================================================
        print("Step 4: Flush telemetry to Collibra")
        print("-" * 70)

        exported = adapter.flush()
        print(f"   [OK] Flushed {exported} assets to Collibra")
        print()

        # ============================================================
        # STEP 5: View in Collibra
        # ============================================================
        print("Step 5: View in Collibra UI")
        print("-" * 70)
        print("   Log into your Collibra instance:")
        print(f"      {os.getenv('COLLIBRA_URL')}")
        print()
        print("   Navigate to your AI Governance domain")
        print()
        print("   View exported assets:")
        print("      • AI Operation Cost assets")
        print("      • Policy Evaluation assets")
        print("      • Complete governance metadata")
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
        print(f"[ERROR] Error during quickstart: {e}")
        import traceback

        traceback.print_exc()
        return False


def print_next_steps() -> None:
    """Print next steps."""
    print()
    print("=" * 70)
    print("Quickstart Complete!")
    print("=" * 70)
    print()
    print("What just happened?")
    print("  1. [OK] Auto-instrumentation enabled Collibra integration")
    print("  2. [OK] Your AI operations automatically exported governance data")
    print("  3. [OK] Cost, policy, and metadata tracked transparently")
    print("  4. [OK] Data now visible in Collibra for audit and compliance")
    print()
    print("Next Steps:")
    print("  1. View your data in Collibra UI")
    print("  2. Explore 02_basic_export.py for manual instrumentation")
    print("  3. Try 03_policy_import.py for bidirectional policy sync")
    print("  4. Integrate into your production AI applications")
    print()
    print("Key Features:")
    print("  • Zero-code integration with auto_instrument()")
    print("  • Batch export for efficiency (100x fewer API calls)")
    print("  • Real-time export for critical events")
    print("  • Budget tracking and alerting")
    print("  • Policy enforcement integration")
    print()
    print("Documentation:")
    print("  • Quickstart: docs/quickstarts/collibra-quickstart.md")
    print("  • Full Guide: docs/integrations/collibra.md")
    print()


def main() -> int:
    """Main quickstart workflow."""
    print_header()

    # Check prerequisites
    if not check_prerequisites():
        print("[ERROR] Prerequisites not met.")
        print()
        print("Quick Fix:")
        print("  1. Set COLLIBRA_URL, COLLIBRA_USERNAME, COLLIBRA_PASSWORD")
        print("  2. (Optional) Set GENOPS_TEAM and GENOPS_PROJECT")
        print("  3. Run again: python 01_quickstart_demo.py")
        return 1

    # Run quickstart
    success = demonstrate_quickstart()

    if success:
        print_next_steps()
        return 0
    else:
        print("=" * 70)
        print("[ERROR] Quickstart failed!")
        print("=" * 70)
        print()
        print("Troubleshooting:")
        print("  1. Verify Collibra credentials are correct")
        print("  2. Check Collibra instance is accessible")
        print("  3. Ensure at least one domain exists in Collibra")
        print("  4. Run validation:")
        print("     python -m genops.providers.collibra.validation")
        print()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
