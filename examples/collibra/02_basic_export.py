#!/usr/bin/env python3
"""
Collibra Basic Export Example

This example demonstrates exporting GenOps governance telemetry to Collibra
as governance assets for auditing and compliance.

Features demonstrated:
- Manual adapter initialization
- AI operation tracking with cost telemetry
- Policy enforcement tracking
- Automatic export to Collibra
- Export statistics and reporting

Prerequisites:
    export COLLIBRA_URL="https://your-instance.collibra.com"
    export COLLIBRA_USERNAME="your-username"
    export COLLIBRA_PASSWORD="your-password"
    # OR use API token:
    export COLLIBRA_API_TOKEN="your-api-token"

Run this example:
    python 02_basic_export.py
"""

import os
import sys
import time


def print_header() -> None:
    """Print example header."""
    print("=" * 70)
    print("GenOps + Collibra: Basic Export Example")
    print("=" * 70)
    print()


def check_prerequisites() -> bool:
    """Check if required dependencies are available."""
    print("Prerequisites Check:")

    missing = []

    # Check GenOps installation
    try:
        import genops  # noqa: F401

        print("  \u2713 GenOps installed")
    except ImportError:
        print("  \u2717 GenOps not installed")
        missing.append("pip install genops")

    # Check Collibra credentials
    has_basic_auth = os.getenv("COLLIBRA_USERNAME") and os.getenv("COLLIBRA_PASSWORD")
    has_token = os.getenv("COLLIBRA_API_TOKEN")

    if os.getenv("COLLIBRA_URL"):
        print("  \u2713 COLLIBRA_URL configured")
    else:
        print("  \u2717 COLLIBRA_URL not set")
        missing.append('export COLLIBRA_URL="https://your-instance.collibra.com"')

    if has_basic_auth or has_token:
        auth_method = "token" if has_token else "basic auth"
        print(f"  \u2713 Collibra credentials configured ({auth_method})")
    else:
        print("  \u2717 Collibra credentials not configured")
        missing.append('export COLLIBRA_USERNAME="your-username"')
        missing.append('export COLLIBRA_PASSWORD="your-password"')

    if missing:
        print("\n\u2717 Missing requirements:")
        for req in missing:
            print(f"   {req}")
        return False

    print("  \u2713 All prerequisites met!")
    print()
    return True


def demonstrate_basic_export() -> int:
    """Demonstrate basic telemetry export to Collibra."""
    print("Demonstrating Basic Export to Collibra...")
    print()

    try:
        from genops.providers.collibra import GenOpsCollibraAdapter

        # Initialize adapter
        print("1. Initializing GenOps Collibra adapter...")
        adapter = GenOpsCollibraAdapter(
            team="ml-platform",
            project="ai-governance-demo",
            environment="development",
            export_mode="batch",  # Batch mode for efficiency
            batch_size=10,
            batch_interval_seconds=30,
            enable_cost_tracking=True,
            daily_budget_limit=100.0,
        )

        print("   \u2713 Adapter initialized")
        print("     - Team: ml-platform")
        print("     - Project: ai-governance-demo")
        print("     - Export mode: batch")
        print("     - Daily budget: $100.00")
        print()

        # Simulate AI operations with telemetry
        print("2. Simulating AI operations with governance telemetry...")

        # Operation 1: Cost tracking
        print("   Operation 1: GPT-4 Completion (cost tracking)")
        with adapter.track_ai_operation(
            "gpt-4-completion", customer_id="enterprise-customer-123"
        ) as span:
            # Simulate AI operation
            time.sleep(0.1)

            # Record cost telemetry
            adapter.record_cost(
                span,
                cost=0.05,
                provider="openai",
                model="gpt-4",
                tokens_input=150,
                tokens_output=200,
            )

        print("     \u2713 Cost: $0.05 (350 tokens)")

        # Operation 2: Policy enforcement
        print("   Operation 2: Claude Completion (policy enforcement)")
        with adapter.track_ai_operation(
            "claude-3-completion", customer_id="startup-customer-456"
        ) as span:
            time.sleep(0.1)

            # Record cost
            adapter.record_cost(
                span, cost=0.03, provider="anthropic", model="claude-3-opus"
            )

            # Record policy enforcement
            adapter.record_policy(
                span,
                policy_name="cost_limit",
                policy_result="allowed",
                policy_reason="Within budget limit",
            )

        print("     \u2713 Cost: $0.03 (policy: allowed)")

        # Operation 3: High-cost operation
        print("   Operation 3: GPT-4-Turbo Batch (high-cost)")
        with adapter.track_ai_operation(
            "gpt-4-turbo-batch", customer_id="enterprise-customer-123"
        ) as span:
            time.sleep(0.1)

            # Record significant cost
            adapter.record_cost(
                span,
                cost=2.50,
                provider="openai",
                model="gpt-4-turbo",
                tokens_input=5000,
                tokens_output=10000,
            )

        print("     \u2713 Cost: $2.50 (15,000 tokens)")
        print()

        # Flush exports to Collibra
        print("3. Flushing telemetry to Collibra...")
        exported_count = adapter.flush()
        print(f"   \u2713 Exported {exported_count} assets to Collibra")
        print()

        # Display metrics
        print("4. Adapter Metrics:")
        metrics = adapter.get_metrics()
        print(f"   Operations tracked: {metrics['operation_count']}")
        print(f"   Total cost: ${metrics['total_cost']:.2f}")
        print(f"   Budget remaining: ${metrics['budget_remaining']:.2f}")
        print(f"   Assets exported: {metrics['assets_exported']}")
        print(f"   Batches sent: {metrics['batches_sent']}")
        print()

        # Display export summary
        print("5. Export Summary:")
        summary = adapter.get_export_summary()
        print(f"   Assets created: {summary['assets_created']}")
        print(f"   Assets failed: {summary['assets_failed']}")
        print(f"   Total cost tracked: ${summary['total_cost']:.2f}")
        avg_time = summary["average_export_time_ms"]
        print(f"   Average export time: {avg_time:.2f}ms")
        print()

        # Shutdown adapter
        print("6. Shutting down adapter...")
        adapter.shutdown()
        print("   \u2713 Adapter shutdown complete")
        print()

        return True

    except ImportError as e:
        print(f"\u2717 Required package not available: {e}")
        print("   Fix: pip install genops")
        return False
    except Exception as e:
        print(f"\u2717 Error during export: {e}")
        import traceback

        traceback.print_exc()
        return False


def print_next_steps() -> None:
    """Print next steps and usage guidance."""
    print()
    print("=" * 70)
    print("Next Steps")
    print("=" * 70)
    print()
    print("1. View Assets in Collibra:")
    print("   - Log into your Collibra instance")
    print("   - Navigate to your AI Governance domain")
    print("   - View exported AI operation assets")
    print()
    print("2. Explore Other Examples:")
    print("   - 01_quickstart_demo.py: Zero-code auto-instrumentation")
    print("   - 03_policy_import.py: Import and enforce Collibra policies")
    print("   - 04_bidirectional_sync.py: Full bidirectional integration")
    print()
    print("3. Integration Modes:")
    print("   - Batch mode: Efficient for high-volume operations")
    print("   - Real-time mode: Immediate export for critical events")
    print("   - Hybrid mode: Best of both worlds")
    print()
    print("4. Documentation:")
    print("   - Quickstart: docs/quickstarts/collibra-quickstart.md")
    print("   - Full guide: docs/integrations/collibra.md")
    print()


def main() -> int:
    """Main example workflow."""
    print_header()

    # Check prerequisites
    if not check_prerequisites():
        print(
            "\u2717 Prerequisites not met. Please install dependencies and set credentials."
        )
        return 1

    # Run basic export demonstration
    success = demonstrate_basic_export()

    if success:
        print("=" * 70)
        print("\u2713 Basic export example completed successfully!")
        print("=" * 70)

        print_next_steps()
        return 0
    else:
        print("=" * 70)
        print("\u2717 Basic export example failed!")
        print("=" * 70)
        print()
        print("Troubleshooting:")
        print("  1. Check Collibra credentials are correct")
        print("  2. Verify Collibra instance URL is accessible")
        print("  3. Ensure you have at least one domain in Collibra")
        print("  4. Run validation: python -m genops.providers.collibra.validation")
        print()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
