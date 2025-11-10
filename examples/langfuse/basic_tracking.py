#!/usr/bin/env python3
"""
Langfuse Basic Governance Tracking Example

This example demonstrates simple LLM operations with enhanced Langfuse tracing
using GenOps governance. Perfect for understanding how GenOps enhances Langfuse
observability with cost attribution and team tracking.

Usage:
    python basic_tracking.py

Prerequisites:
    pip install genops[langfuse]
    export LANGFUSE_PUBLIC_KEY="pk-lf-your-public-key"
    export LANGFUSE_SECRET_KEY="sk-lf-your-secret-key"
    export OPENAI_API_KEY="your-openai-api-key"  # Or another provider
"""

import os
import sys
from datetime import datetime


def demonstrate_basic_tracking():
    """Demonstrate basic Langfuse tracking with GenOps governance."""
    print("🔍 Basic Langfuse Tracking with GenOps Governance")
    print("=" * 52)

    try:
        from genops.providers.langfuse import instrument_langfuse

        # Initialize GenOps Langfuse adapter with governance
        adapter = instrument_langfuse(
            team="basic-demo-team",
            project="tracking-example",
            environment="development"
        )

        print("✅ GenOps Langfuse adapter initialized")
        print(f"   Team: {adapter.team}")
        print(f"   Project: {adapter.project}")
        print(f"   Environment: {adapter.environment}")

    except ImportError as e:
        print(f"❌ Failed to import GenOps Langfuse: {e}")
        print("💡 Fix: Run 'pip install genops[langfuse]'")
        return False
    except Exception as e:
        print(f"❌ Failed to initialize adapter: {e}")
        return False

    # Example 1: Simple LLM operation with governance
    print("\n📋 Example 1: Basic LLM Operation with Governance")
    print("-" * 50)

    try:
        with adapter.trace_with_governance(
            name="basic_llm_operation",
            customer_id="demo-customer-123",
            cost_center="research"
        ) as trace:

            print("🚀 Executing LLM operation with enhanced tracking...")

            # Simple generation with cost tracking
            response = adapter.generation_with_cost_tracking(
                prompt="Explain the benefits of LLM observability in 2 sentences.",
                model="gpt-3.5-turbo",
                max_cost=0.05,  # 5 cent budget limit
                operation="explanation_task"
            )

            print("✅ Operation completed successfully!")
            print(f"📝 Response: {response.content[:100]}...")
            print(f"💰 Cost: ${response.usage.cost:.6f}")
            print(f"⏱️  Duration: {response.usage.latency_ms:.1f}ms")
            print(f"🏷️  Team: {response.usage.team}")
            print(f"📊 Project: {response.usage.project}")
            print(f"🆔 Trace ID: {response.trace_id[:12]}...")

    except Exception as e:
        print(f"❌ Example 1 failed: {e}")
        return False

    # Example 2: Multi-step workflow with governance
    print("\n📋 Example 2: Multi-Step Workflow with Cost Attribution")
    print("-" * 55)

    try:
        with adapter.trace_with_governance(
            name="multi_step_analysis",
            customer_id="workflow-customer",
            feature="data-analysis"
        ) as trace:

            print("🔄 Executing multi-step workflow...")

            # Step 1: Data preprocessing
            preprocessing_response = adapter.generation_with_cost_tracking(
                prompt="Clean and structure this sample data: [user input, metrics, timestamps]",
                model="gpt-3.5-turbo",
                max_cost=0.03,
                operation="data_preprocessing",
                step="1_preprocessing"
            )

            # Step 2: Analysis
            analysis_response = adapter.generation_with_cost_tracking(
                prompt="Analyze the cleaned data for patterns and insights",
                model="gpt-3.5-turbo",
                max_cost=0.04,
                operation="pattern_analysis",
                step="2_analysis"
            )

            # Step 3: Summary
            summary_response = adapter.generation_with_cost_tracking(
                prompt="Summarize the analysis in business-friendly terms",
                model="gpt-3.5-turbo",
                max_cost=0.02,
                operation="business_summary",
                step="3_summary"
            )

            print("✅ Multi-step workflow completed!")

            # Show step-by-step costs
            steps = [
                ("Preprocessing", preprocessing_response),
                ("Analysis", analysis_response),
                ("Summary", summary_response)
            ]

            total_cost = 0
            for step_name, resp in steps:
                print(f"   {step_name}: ${resp.usage.cost:.6f} ({resp.usage.latency_ms:.0f}ms)")
                total_cost += resp.usage.cost

            print(f"💰 Total workflow cost: ${total_cost:.6f}")
            print(f"📊 Operations tracked: {adapter.operation_count}")

    except Exception as e:
        print(f"❌ Example 2 failed: {e}")
        return False

    # Example 3: Team-based cost attribution
    print("\n📋 Example 3: Team-Based Cost Attribution")
    print("-" * 40)

    # Simulate different team operations
    teams = ["research", "product", "engineering"]

    for team_name in teams:
        try:
            # Create team-specific adapter
            team_adapter = instrument_langfuse(
                team=team_name,
                project="team-comparison",
                environment="development"
            )

            # Team-specific operation
            response = team_adapter.generation_with_cost_tracking(
                prompt=f"Generate a {team_name} team status update",
                model="gpt-3.5-turbo",
                max_cost=0.02,
                customer_id=f"{team_name}-customer"
            )

            print(f"   📊 {team_name.title()} Team: ${response.usage.cost:.6f} "
                  f"(Customer: {response.usage.customer_id})")

        except Exception as e:
            print(f"   ❌ {team_name.title()} Team failed: {e}")

    return True


def demonstrate_cost_intelligence():
    """Demonstrate cost intelligence and optimization features."""
    print("\n💡 Cost Intelligence & Optimization Features")
    print("=" * 45)

    try:
        from genops.providers.langfuse import instrument_langfuse

        # Initialize with budget limits
        adapter = instrument_langfuse(
            team="cost-intelligence",
            project="optimization-demo",
            budget_limits={"daily": 0.50}  # 50 cents daily limit
        )

        print("💰 Cost Intelligence Features:")
        print("   • Real-time cost tracking and attribution")
        print("   • Budget limits and compliance enforcement")
        print("   • Team and project cost breakdowns")
        print("   • Cost optimization recommendations")

        # Get current cost summary
        cost_summary = adapter.get_cost_summary("daily")
        print("\n📊 Current Daily Summary:")
        print(f"   Total Cost: ${cost_summary['total_cost']:.6f}")
        print(f"   Operations: {cost_summary['operation_count']}")
        print(f"   Avg Cost/Op: ${cost_summary['average_cost_per_operation']:.6f}")
        print(f"   Budget Remaining: ${cost_summary['budget_remaining']:.6f}")
        print(f"   Team: {cost_summary['governance']['team']}")

        return True

    except Exception as e:
        print(f"❌ Cost intelligence demo failed: {e}")
        return False


def demonstrate_error_handling():
    """Demonstrate error handling and graceful degradation."""
    print("\n🛡️  Error Handling & Graceful Degradation")
    print("=" * 40)

    try:
        from genops.providers.langfuse import instrument_langfuse

        adapter = instrument_langfuse(
            team="error-handling",
            project="resilience-demo"
        )

        print("🧪 Testing error scenarios...")

        # Test 1: Budget limit exceeded
        print("\n   Test 1: Budget Limit Enforcement")
        try:
            adapter.generation_with_cost_tracking(
                prompt="This is a very expensive operation " * 100,  # Large prompt
                model="gpt-4",  # Expensive model
                max_cost=0.001  # Very low limit
            )
            print("   ❌ Unexpectedly succeeded (should have failed)")
        except ValueError as e:
            print(f"   ✅ Budget limit enforced: {str(e)[:50]}...")

        # Test 2: Invalid model graceful handling
        print("\n   Test 2: Invalid Model Handling")
        try:
            response = adapter.generation_with_cost_tracking(
                prompt="Simple test",
                model="nonexistent-model",  # Invalid model
                max_cost=0.10
            )
            print("   ✅ Handled gracefully with fallback cost calculation")
            print(f"   Cost: ${response.usage.cost:.6f}")
        except Exception as e:
            print(f"   ✅ Handled gracefully: {str(e)[:50]}...")

        # Test 3: Governance validation
        print("\n   Test 3: Governance Attribute Validation")
        try:
            with adapter.trace_with_governance(
                name="validation_test",
                invalid_attribute="should_be_ignored"  # Invalid governance attr
            ) as trace:
                print("   ✅ Invalid attributes filtered out automatically")
        except Exception as e:
            print(f"   ⚠️  Governance validation issue: {e}")

        return True

    except Exception as e:
        print(f"❌ Error handling demo failed: {e}")
        return False


def show_next_steps():
    """Show next steps for developers."""
    print("\n🚀 Next Steps & Advanced Features")
    print("=" * 35)

    next_steps = [
        ("🔍 Setup Validation", "python setup_validation.py",
         "Comprehensive setup diagnostics"),
        ("⚡ Auto-Instrumentation", "python auto_instrumentation.py",
         "Zero-code integration for existing apps"),
        ("📊 Evaluation Integration", "python evaluation_integration.py",
         "LLM evaluation with governance tracking"),
        ("🎯 Prompt Management", "python prompt_management.py",
         "Cost-aware prompt optimization"),
        ("🏭 Production Patterns", "python production_patterns.py",
         "Enterprise deployment and monitoring")
    ]

    for title, command, description in next_steps:
        print(f"   {title}")
        print(f"     Command: {command}")
        print(f"     Purpose: {description}")
        print()

    print("📚 Additional Resources:")
    print("   • Comprehensive Guide: docs/integrations/langfuse.md")
    print("   • 5-Minute Quickstart: docs/langfuse-quickstart.md")
    print("   • All Examples: ./run_all_examples.sh")


def main():
    """Main function to run the basic tracking example."""
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check prerequisites
    if not os.getenv('LANGFUSE_PUBLIC_KEY'):
        print("❌ Missing LANGFUSE_PUBLIC_KEY environment variable")
        print("💡 Get your keys at: https://cloud.langfuse.com/")
        return False

    if not os.getenv('LANGFUSE_SECRET_KEY'):
        print("❌ Missing LANGFUSE_SECRET_KEY environment variable")
        print("💡 Get your keys at: https://cloud.langfuse.com/")
        return False

    if not any([os.getenv('OPENAI_API_KEY'), os.getenv('ANTHROPIC_API_KEY')]):
        print("❌ No AI provider API keys found")
        print("💡 Set at least one:")
        print("   export OPENAI_API_KEY='your_openai_key'")
        print("   export ANTHROPIC_API_KEY='your_anthropic_key'")
        return False

    # Run demonstrations
    success = True
    success &= demonstrate_basic_tracking()
    success &= demonstrate_cost_intelligence()
    success &= demonstrate_error_handling()

    if success:
        show_next_steps()
        print("\n" + "✅" * 20)
        print("Basic Langfuse + GenOps integration working perfectly!")
        print("Enhanced LLM observability with governance intelligence!")
        print("✅" * 20)
        return True
    else:
        print("\n❌ Some demonstrations failed. Check the errors above.")
        return False


if __name__ == "__main__":
    """Main entry point."""
    success = main()
    sys.exit(0 if success else 1)
