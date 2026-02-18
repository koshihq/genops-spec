#!/usr/bin/env python3
"""
Basic OpenAI Tracking Example

This example demonstrates the simplest way to add GenOps governance telemetry
to your existing OpenAI applications with minimal code changes.

What you'll learn:
- Manual instrumentation with governance attributes
- Cost and performance tracking for chat completions
- Basic error handling and telemetry export

Usage:
    python basic_tracking.py

Prerequisites:
    pip install genops-ai[openai]
    export OPENAI_API_KEY="your_api_key_here"
"""

import os
import sys
import time


def basic_chat_completion():
    """Basic chat completion with GenOps governance tracking."""
    print("💬 Basic Chat Completion with GenOps Tracking")
    print("-" * 50)

    try:
        # Import GenOps OpenAI adapter
        from genops.providers.openai import instrument_openai

        # Create instrumented OpenAI client
        client = instrument_openai(api_key=os.getenv("OPENAI_API_KEY"))
        print("✅ Created instrumented OpenAI client")

        # Make a basic completion with governance attributes
        print("\n🚀 Making OpenAI completion request...")

        response = client.chat_completions_create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "What is artificial intelligence in one sentence?",
                },
            ],
            max_tokens=100,
            temperature=0.7,
            # 🏷️ Governance attributes for cost attribution and tracking
            team="ai-examples",
            project="genops-demo",
            customer_id="demo-user-001",
            environment="development",
            feature="basic-tracking",
        )

        # Display results
        print("✅ Request completed successfully!")
        print(f"\n📝 Response: {response.choices[0].message.content}")
        print("\n📊 Usage Stats:")
        print(f"   • Input tokens: {response.usage.prompt_tokens}")
        print(f"   • Output tokens: {response.usage.completion_tokens}")
        print(f"   • Total tokens: {response.usage.total_tokens}")

        # The cost and governance attributes are automatically tracked
        # and exported to your configured observability platform
        print("\n💰 Cost tracking: Automatically calculated and exported")
        print("🏷️  Governance: Attributed to team 'ai-examples', project 'genops-demo'")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Fix: Run 'pip install genops-ai[openai]'")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Fix: Check your OPENAI_API_KEY and network connectivity")
        return False


def batch_processing_example():
    """Example of tracking costs across multiple OpenAI operations."""
    print("\n\n📦 Batch Processing with Cost Aggregation")
    print("-" * 50)

    try:
        from genops import track
        from genops.providers.openai import instrument_openai

        client = instrument_openai()

        # Sample tasks to process
        tasks = [
            "Summarize: AI is transforming how we work and live.",
            "Translate to French: Hello, how are you today?",
            "Generate a creative name for a coffee shop.",
        ]

        # Use context manager to track batch operation costs
        with track(
            "batch_processing",
            team="batch-team",
            project="multi-task-demo",
            customer_id="batch-customer-001",
        ) as span:
            results = []
            total_tokens = 0

            print("🔄 Processing tasks...")
            for i, task in enumerate(tasks):
                print(f"   Task {i + 1}: {task[:30]}...")

                response = client.chat_completions_create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": task}],
                    max_tokens=50,
                    # Individual task attribution
                    team="batch-team",
                    project="multi-task-demo",
                    customer_id="batch-customer-001",
                    task_index=i,
                    batch_id="demo-batch-001",
                )

                results.append(response.choices[0].message.content.strip())
                total_tokens += response.usage.total_tokens

                # Brief pause between requests
                time.sleep(0.5)

            # Set batch-level attributes
            span.set_attribute("tasks_processed", len(tasks))
            span.set_attribute("total_tokens", total_tokens)

            print("\n✅ Batch completed!")
            print("📊 Results:")
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result}")

            print(f"\n💰 Total tokens across batch: {total_tokens}")
            print("🏷️  Costs automatically attributed to 'batch-team' project")

        return True

    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        return False


def governance_attributes_demo():
    """Demonstrate different governance attribute patterns."""
    print("\n\n🏷️  Governance Attributes Demo")
    print("-" * 50)

    try:
        from genops.providers.openai import instrument_openai

        client = instrument_openai()

        # Example 1: Customer support use case
        print("📞 Customer Support Scenario:")
        support_response = client.chat_completions_create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "How do I reset my password?"}],
            max_tokens=100,
            # Customer support governance attributes
            team="customer-support",
            project="help-desk-automation",
            customer_id="customer-12345",
            environment="production",
            cost_center="support-operations",
            feature="password-reset-help",
        )
        print(f"   Response: {support_response.choices[0].message.content[:60]}...")

        # Example 2: Product development use case
        print("\n🛠️  Product Development Scenario:")
        dev_response = client.chat_completions_create(
            model="gpt-4",  # Using more powerful model for complex tasks
            messages=[
                {
                    "role": "user",
                    "content": "Review this code for security issues: function login(user, pass) { return user === 'admin' && pass === '123'; }",
                }
            ],
            max_tokens=150,
            # Development team governance attributes
            team="engineering",
            project="security-review-automation",
            environment="development",
            cost_center="rd-department",
            feature="code-security-analysis",
            user_id="developer-789",
        )
        print(f"   Response: {dev_response.choices[0].message.content[:60]}...")

        print("\n💡 Each request is attributed to its respective team and project")
        print("📊 This enables detailed cost allocation and usage analytics")

        return True

    except Exception as e:
        print(f"❌ Governance demo error: {e}")
        return False


def main():
    """Run all basic tracking examples."""
    print("🚀 GenOps + OpenAI Basic Tracking Examples")
    print("=" * 60)

    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("💡 Fix: export OPENAI_API_KEY='your_api_key_here'")
        return False

    success = True

    # Run examples
    success &= basic_chat_completion()
    success &= batch_processing_example()
    success &= governance_attributes_demo()

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("🎉 All basic tracking examples completed successfully!")
        print("\n📚 What happened:")
        print(
            "   • OpenAI requests were automatically instrumented with GenOps telemetry"
        )
        print("   • Costs were calculated and attributed to teams/projects/customers")
        print("   • Governance attributes enable detailed cost allocation")
        print("   • All telemetry was exported to your observability platform")

        print("\n🚀 Next steps:")
        print("   • Run 'python auto_instrumentation.py' for zero-code setup")
        print("   • Try 'python cost_optimization.py' for multi-model cost analysis")
        print(
            "   • Explore 'python advanced_features.py' for streaming, functions, etc."
        )

        return True
    else:
        print("❌ Some examples failed. Check the error messages above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
