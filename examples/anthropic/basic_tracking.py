#!/usr/bin/env python3
"""
Basic Anthropic Tracking Example

This example demonstrates the simplest way to add GenOps governance telemetry
to your existing Anthropic Claude applications with minimal code changes.

What you'll learn:
- Manual instrumentation with governance attributes
- Cost and performance tracking for Claude messages
- Basic error handling and telemetry export

Usage:
    python basic_tracking.py

Prerequisites:
    pip install genops-ai[anthropic]
    export ANTHROPIC_API_KEY="your_anthropic_key_here"
"""

import os
import sys
import time


def basic_message_creation():
    """Basic Claude message creation with GenOps governance tracking."""
    print("💬 Basic Claude Message with GenOps Tracking")
    print("-" * 50)

    try:
        # Import GenOps Anthropic adapter
        from genops.providers.anthropic import instrument_anthropic

        # Create instrumented Anthropic client
        client = instrument_anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        print("✅ Created instrumented Anthropic client")

        # Make a basic message with governance attributes
        print("\n🚀 Making Claude message request...")

        response = client.messages_create(
            model="claude-3-5-sonnet-20241022",
            messages=[
                {"role": "user", "content": "Explain artificial intelligence in one clear paragraph."}
            ],
            max_tokens=150,
            temperature=0.7,

            # 🏷️ Governance attributes for cost attribution and tracking
            team="ai-examples",
            project="genops-demo",
            customer_id="demo-user-001",
            environment="development",
            feature="basic-tracking"
        )

        # Display results
        print("✅ Request completed successfully!")
        print(f"\n📝 Response: {response.content[0].text}")
        print("\n📊 Usage Stats:")
        print(f"   • Input tokens: {response.usage.input_tokens}")
        print(f"   • Output tokens: {response.usage.output_tokens}")
        print(f"   • Total tokens: {response.usage.input_tokens + response.usage.output_tokens}")

        # The cost and governance attributes are automatically tracked
        # and exported to your configured observability platform
        print("\n💰 Cost tracking: Automatically calculated and exported")
        print("🏷️  Governance: Attributed to team 'ai-examples', project 'genops-demo'")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Fix: Run 'pip install genops-ai[anthropic]'")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Fix: Check your ANTHROPIC_API_KEY and network connectivity")
        return False

def batch_processing_example():
    """Example of tracking costs across multiple Claude operations."""
    print("\n\n📦 Batch Processing with Cost Aggregation")
    print("-" * 50)

    try:
        from genops import track
        from genops.providers.anthropic import instrument_anthropic

        client = instrument_anthropic()

        # Sample tasks to process
        tasks = [
            "Summarize the benefits of renewable energy in 2 sentences.",
            "Explain machine learning to a 10-year-old in simple terms.",
            "What are the top 3 programming languages for data science?"
        ]

        # Use context manager to track batch operation costs
        with track("batch_processing",
                   team="batch-team",
                   project="multi-task-demo",
                   customer_id="batch-customer-001") as span:

            results = []
            total_tokens = 0

            print("🔄 Processing tasks...")
            for i, task in enumerate(tasks):
                print(f"   Task {i+1}: {task[:40]}...")

                response = client.messages_create(
                    model="claude-3-haiku-20240307",  # Fast and cost-effective for batch
                    messages=[{"role": "user", "content": task}],
                    max_tokens=100,

                    # Individual task attribution
                    team="batch-team",
                    project="multi-task-demo",
                    customer_id="batch-customer-001",
                    task_index=i,
                    batch_id="demo-batch-001"
                )

                results.append(response.content[0].text.strip())
                total_tokens += response.usage.input_tokens + response.usage.output_tokens

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
        from genops.providers.anthropic import instrument_anthropic

        client = instrument_anthropic()

        # Example 1: Legal document analysis
        print("⚖️  Legal Analysis Scenario:")
        legal_response = client.messages_create(
            model="claude-3-5-sonnet-20241022",  # High-quality for legal work
            messages=[{"role": "user", "content": "What are the key elements of a software license agreement?"}],
            max_tokens=200,

            # Legal department governance attributes
            team="legal-team",
            project="contract-analysis-automation",
            customer_id="internal-legal-dept",
            environment="production",
            cost_center="legal-operations",
            feature="license-analysis",
            requires_expertise="legal"
        )
        print(f"   Response: {legal_response.content[0].text[:80]}...")

        # Example 2: Content creation use case
        print("\n✍️  Content Creation Scenario:")
        content_response = client.messages_create(
            model="claude-3-5-haiku-20241022",  # Fast for content generation
            messages=[{"role": "user", "content": "Write a compelling headline for a blog post about sustainable technology."}],
            max_tokens=50,

            # Content team governance attributes
            team="content-marketing",
            project="blog-automation",
            environment="development",
            cost_center="marketing-department",
            feature="headline-generation",
            user_id="content-creator-123"
        )
        print(f"   Response: {content_response.content[0].text[:80]}...")

        # Example 3: Customer service automation
        print("\n🎧 Customer Service Scenario:")
        service_response = client.messages_create(
            model="claude-3-5-sonnet-20241022",  # Balanced for customer interactions
            messages=[
                {"role": "user", "content": "How do I reset my password and ensure my account is secure?"}
            ],
            max_tokens=150,

            # Customer service governance attributes
            team="customer-support",
            project="automated-help-desk",
            customer_id="customer-service-bot",
            environment="production",
            cost_center="support-operations",
            feature="password-help",
            conversation_type="support_chat"
        )
        print(f"   Response: {service_response.content[0].text[:80]}...")

        print("\n💡 Each request is attributed to its respective team and project")
        print("📊 This enables detailed cost allocation and usage analytics")

        return True

    except Exception as e:
        print(f"❌ Governance demo error: {e}")
        return False

def claude_model_comparison():
    """Compare different Claude models for the same task."""
    print("\n\n🤖 Claude Model Comparison")
    print("-" * 50)

    try:
        from genops.providers.anthropic import instrument_anthropic

        client = instrument_anthropic()

        test_prompt = "Explain the concept of machine learning and its applications in healthcare."

        # Test different Claude models
        models_to_test = [
            {
                "name": "claude-3-haiku-20240307",
                "description": "Fast and cost-effective",
                "use_case": "High-volume, simple tasks"
            },
            {
                "name": "claude-3-5-haiku-20241022",
                "description": "Balanced speed and capability",
                "use_case": "General purpose applications"
            },
            {
                "name": "claude-3-5-sonnet-20241022",
                "description": "Advanced reasoning and analysis",
                "use_case": "Complex tasks requiring nuanced understanding"
            }
        ]

        print(f"📝 Test prompt: {test_prompt}")
        print("\n📊 Model Comparison Results:")
        print(f"{'Model':<30} {'Tokens':<10} {'Response Preview'}")
        print("-" * 80)

        for model_config in models_to_test:
            try:
                print(f"🔄 Testing {model_config['name'][:20]}...")

                response = client.messages_create(
                    model=model_config["name"],
                    messages=[{"role": "user", "content": test_prompt}],
                    max_tokens=200,  # Fixed for fair comparison
                    temperature=0.7,

                    # Model comparison tracking
                    team="comparison-team",
                    project="model-evaluation",
                    customer_id="model-comparison-demo",
                    model_test=model_config["name"],
                    comparison_study="claude_models",
                    use_case=model_config["use_case"]
                )

                total_tokens = response.usage.input_tokens + response.usage.output_tokens
                response_preview = response.content[0].text[:50] + "..."

                print(f"{model_config['name']:<30} {total_tokens:<10} {response_preview}")

            except Exception as e:
                print(f"{model_config['name']:<30} Error: {str(e)[:30]}...")

        print("\n💡 Model Selection Guidelines:")
        for model_config in models_to_test:
            print(f"   • {model_config['name'][:25]}: {model_config['use_case']}")

        return True

    except Exception as e:
        print(f"❌ Model comparison error: {e}")
        return False

def main():
    """Run all basic tracking examples."""
    print("🚀 GenOps + Anthropic Basic Tracking Examples")
    print("=" * 60)

    # Check prerequisites
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY environment variable not set")
        print("💡 Fix: export ANTHROPIC_API_KEY='your_api_key_here'")
        return False

    success = True

    # Run examples
    success &= basic_message_creation()
    success &= batch_processing_example()
    success &= governance_attributes_demo()
    success &= claude_model_comparison()

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("🎉 All basic tracking examples completed successfully!")
        print("\n📚 What happened:")
        print("   • Anthropic Claude requests were automatically instrumented with GenOps telemetry")
        print("   • Costs were calculated and attributed to teams/projects/customers")
        print("   • Governance attributes enable detailed cost allocation")
        print("   • All telemetry was exported to your observability platform")

        print("\n🚀 Next steps:")
        print("   • Run 'python auto_instrumentation.py' for zero-code setup")
        print("   • Try 'python cost_optimization.py' for Claude model optimization")
        print("   • Explore 'python advanced_features.py' for streaming, conversations, etc.")

        return True
    else:
        print("❌ Some examples failed. Check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
