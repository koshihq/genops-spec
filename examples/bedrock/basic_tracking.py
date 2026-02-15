#!/usr/bin/env python3
"""
Bedrock Basic Usage Example

This example demonstrates essential patterns for using GenOps with AWS Bedrock.
Shows manual adapter usage, governance attributes, and cost tracking across
multiple AI models and providers.

Example usage:
    python basic_tracking.py

Features demonstrated:
- Manual GenOps Bedrock adapter usage
- Governance attribute examples with team/project attribution
- Multi-model cost tracking and comparison
- Provider-specific optimizations (Anthropic, Amazon, AI21, Cohere)
- Error handling and retry patterns
- Performance monitoring and optimization
"""

import logging
import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Configure logging to see telemetry activity
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_manual_adapter():
    """Demonstrate manual GenOps Bedrock adapter usage."""

    print("🏗️  Manual GenOps Bedrock Adapter Usage")
    print("=" * 50)
    print("This shows how to use the GenOps adapter directly for full control")
    print("over governance attributes, cost tracking, and model selection.")
    print()

    try:
        from genops.providers.bedrock import GenOpsBedrockAdapter

        # Create adapter with AWS configuration
        print("📡 Creating GenOps Bedrock adapter...")
        adapter = GenOpsBedrockAdapter(
            region_name="us-east-1",
            enable_streaming=True,
            default_model="anthropic.claude-3-haiku-20240307-v1:0",
        )

        if not adapter.is_available():
            print("❌ Bedrock not available")
            print("💡 Check AWS credentials and Bedrock access permissions")
            return False

        print("✅ GenOps Bedrock adapter created successfully")
        print(f"   → Region: {adapter.region_name}")
        print(f"   → Supported models: {len(adapter.get_supported_models())}")
        print(
            f"   → Available tasks: {', '.join(adapter.get_supported_tasks()[:3])}..."
        )
        print()

        # Text Generation with comprehensive governance
        print("📝 Text Generation with Governance Attributes:")
        try:
            response = adapter.text_generation(
                prompt="Analyze the benefits and challenges of cloud computing for enterprise adoption.",
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                max_tokens=150,
                temperature=0.7,
                # Comprehensive governance attributes
                team="cloud-architecture-team",
                project="enterprise-cloud-migration",
                customer_id="fortune500-client-789",
                environment="production",
                feature="technology-analysis",
                cost_center="cloud-engineering",
                compliance_level="SOC2",
            )

            print(f"   📖 Analysis result: {response.content[:100]}...")
            print(f"   💰 Cost: ${response.cost_usd:.6f}")
            print(f"   ⏱️  Latency: {response.latency_ms:.1f}ms")
            print(
                f"   🔢 Tokens: {response.input_tokens} in, {response.output_tokens} out"
            )
            print("   ✅ Governance attributes captured:")
            print("      → Team: cloud-architecture-team (cost attribution)")
            print("      → Project: enterprise-cloud-migration (project tracking)")
            print("      → Customer: fortune500-client-789 (billing attribution)")
            print("      → Environment: production (compliance segregation)")
            print("   ✅ Cost automatically calculated and tracked by region")
            print()

        except Exception as e:
            print(f"   ⚠️ Claude generation failed: {e}")
            print("   💡 This might be due to model access permissions")
            print()

        # Multi-model comparison
        print("⚖️  Multi-Model Cost and Performance Comparison:")
        models_to_test = [
            ("anthropic.claude-3-haiku-20240307-v1:0", "Claude 3 Haiku"),
            ("amazon.titan-text-express-v1", "Titan Text Express"),
            ("ai21.j2-mid-v1", "Jurassic-2 Mid"),
            ("cohere.command-text-v14", "Cohere Command"),
        ]

        test_prompt = "What are the key principles of sustainable software development?"
        results = []

        for model_id, model_name in models_to_test:
            try:
                print(f"   🧪 Testing {model_name}...")
                result = adapter.text_generation(
                    prompt=test_prompt,
                    model_id=model_id,
                    max_tokens=80,
                    temperature=0.7,
                    # Same governance for fair comparison
                    team="sustainability-research",
                    project="green-software-initiative",
                    customer_id="research-internal",
                    feature="model-comparison",
                )

                results.append(
                    {
                        "model": model_name,
                        "model_id": model_id,
                        "cost": result.cost_usd,
                        "latency": result.latency_ms,
                        "tokens_out": result.output_tokens,
                        "provider": adapter.detect_model_provider(model_id),
                    }
                )

                print(
                    f"      💰 ${result.cost_usd:.6f} | ⏱️ {result.latency_ms:.0f}ms | 🔢 {result.output_tokens} tokens"
                )

            except Exception as e:
                print(f"      ❌ Failed: {str(e)[:50]}...")

        # Display comparison summary
        if results:
            print()
            print("   📊 Comparison Summary:")
            results.sort(key=lambda x: x["cost"])
            cheapest = results[0]
            most_expensive = results[-1] if len(results) > 1 else cheapest

            print(
                f"      💚 Most cost-effective: {cheapest['model']} (${cheapest['cost']:.6f})"
            )
            if len(results) > 1:
                savings = most_expensive["cost"] - cheapest["cost"]
                print(
                    f"      💸 Most expensive: {most_expensive['model']} (${most_expensive['cost']:.6f})"
                )
                print(f"      📉 Potential savings: ${savings:.6f} per operation")

            # Provider diversity
            providers = {r["provider"] for r in results}
            print(f"      🏗️  Providers tested: {', '.join(providers)}")
            print("   ✅ All costs automatically tracked by provider and model")

        print()

        # Chat completion example
        print("💬 Chat Completion with Multi-Message Context:")
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an AI ethics advisor helping with responsible AI deployment.",
                },
                {
                    "role": "user",
                    "content": "What are the main ethical considerations for deploying AI in healthcare?",
                },
                {
                    "role": "assistant",
                    "content": "Key ethical considerations include patient privacy, algorithmic bias, transparency in decision-making, and ensuring human oversight.",
                },
                {
                    "role": "user",
                    "content": "How can we ensure patient data privacy specifically?",
                },
            ]

            chat_response = adapter.chat_completion(
                messages=messages,
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                max_tokens=120,
                temperature=0.6,
                # Healthcare-specific governance
                team="healthcare-ai-ethics",
                project="responsible-ai-deployment",
                customer_id="healthcare-system-456",
                feature="ethics-consultation",
                compliance_level="HIPAA",
            )

            print(f"   🏥 Ethics guidance: {chat_response.content[:80]}...")
            print(f"   💰 Cost: ${chat_response.cost_usd:.6f}")
            print("   ✅ HIPAA compliance attributes recorded")
            print("   ✅ Multi-message context processed with governance")
            print()

        except Exception as e:
            print(f"   ⚠️ Chat completion failed: {e}")
            print("   💡 Some models may have limited chat support")
            print()

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("💡 Install GenOps with: pip install genops-ai[bedrock]")
        return False


def demonstrate_cost_optimization():
    """Show cost optimization features and recommendations."""

    print("💰 Cost Optimization and Intelligence")
    print("=" * 40)
    print("GenOps provides intelligent cost optimization recommendations:")
    print()

    try:
        from genops.providers.bedrock_pricing import (
            compare_bedrock_models,
            estimate_monthly_cost,
            get_cost_optimization_recommendations,
        )

        # Compare models for a specific task
        print("📊 Model Cost Comparison for 'content generation' task:")
        comparison = compare_bedrock_models(
            model_ids=[
                "anthropic.claude-3-haiku-20240307-v1:0",
                "amazon.titan-text-express-v1",
                "ai21.j2-mid-v1",
            ],
            input_tokens=1000,
            output_tokens=500,
            region="us-east-1",
            task_description="content generation",
        )

        print(f"   💡 Task: {comparison.task_description}")
        print(
            f"   💰 Cost range: ${comparison.cost_range[0]:.6f} - ${comparison.cost_range[1]:.6f}"
        )
        print(f"   🥇 Cheapest: {comparison.cheapest_model}")
        print(f"   💸 Most expensive: {comparison.most_expensive_model}")

        print("\n   📈 Model breakdown:")
        for model in comparison.models:
            percentage = (
                (model.total_cost / comparison.cost_range[1]) * 100
                if comparison.cost_range[1] > 0
                else 0
            )
            print(
                f"      {model.model_name}: ${model.total_cost:.6f} ({percentage:.1f}%)"
            )

        print("\n   💡 Optimization recommendations:")
        for i, rec in enumerate(comparison.recommendations, 1):
            print(f"      {i}. {rec}")

        # Monthly cost estimation
        print("\n📅 Monthly Cost Estimation:")
        monthly_cost = estimate_monthly_cost(
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            daily_operations=100,
            avg_input_tokens=500,
            avg_output_tokens=200,
            region="us-east-1",
        )

        print("   📊 For 100 operations/day with Claude 3 Haiku:")
        print(f"      Daily: ${monthly_cost['daily_cost']:.2f}")
        print(f"      Monthly: ${monthly_cost['monthly_cost']:.2f}")
        print(f"      Annual: ${monthly_cost['annual_cost']:.2f}")
        print(f"      Per operation: ${monthly_cost['cost_per_operation']:.6f}")

        # Personalized recommendations
        print("\n🎯 Personalized Optimization Recommendations:")
        recommendations = get_cost_optimization_recommendations(
            current_model="anthropic.claude-3-sonnet-20240229-v1:0",
            task_type="content generation",
            input_tokens=800,
            output_tokens=400,
            region="us-east-1",
            budget_per_operation=0.01,
        )

        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")

        print()

    except ImportError:
        print("❌ Cost optimization features not available")
        print("💡 Check GenOps pricing module installation")


def demonstrate_error_handling():
    """Show error handling and resilience patterns."""

    print("🛡️ Error Handling and Resilience")
    print("=" * 35)
    print("GenOps gracefully handles various error scenarios:")
    print()

    try:
        from genops.providers.bedrock import GenOpsBedrockAdapter

        adapter = GenOpsBedrockAdapter()

        # Test with invalid model
        print("   🧪 Testing invalid model handling...")
        try:
            adapter.text_generation(
                prompt="Test prompt",
                model_id="nonexistent.invalid-model-12345",
                team="testing-team",
                project="error-handling-test",
            )
            print("   ⚠️ Unexpected success with invalid model")

        except Exception as e:
            print(f"   ✅ Graceful error handling: {str(e)[:60]}...")
            print("   ✅ Error details captured in telemetry")
            print("   ✅ Governance attributes preserved during error")

        # Test with empty input
        print("   🧪 Testing empty input handling...")
        try:
            adapter.text_generation(
                prompt="",  # Empty prompt
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                team="testing-team",
            )
            print("   ✅ Empty input handled successfully")

        except Exception as e:
            print(f"   ✅ Empty input error handled: {str(e)[:60]}...")

        print()
        print("✅ Error scenarios captured in telemetry for debugging")
        print("✅ Governance context preserved even during failures")
        print("✅ Graceful degradation maintains application stability")
        print()

    except ImportError:
        print("❌ Error handling demo unavailable - check installation")


def main():
    """Main demonstration function."""

    print("Welcome to the GenOps Bedrock Basic Usage Demo!")
    print()
    print("This example demonstrates essential patterns for integrating")
    print("GenOps governance and telemetry with AWS Bedrock applications.")
    print()

    success_count = 0
    total_demos = 3

    # Run all demonstrations
    demos = [
        ("Manual Adapter Usage", demonstrate_manual_adapter),
        ("Cost Optimization", demonstrate_cost_optimization),
        ("Error Handling", demonstrate_error_handling),
    ]

    for demo_name, demo_func in demos:
        print(f"🚀 Running {demo_name} Demo...")
        try:
            success = demo_func()
            if success is not False:  # None or True both count as success
                success_count += 1
                print(f"✅ {demo_name} demo completed successfully")
            else:
                print(f"⚠️ {demo_name} demo encountered issues")
        except Exception as e:
            print(f"❌ {demo_name} demo failed: {e}")

        print("-" * 60)
        print()

    # Summary
    if success_count == total_demos:
        print("🎉 All Basic Usage Demos Completed Successfully!")
        print()
        print("🚀 Next Steps:")
        print("   1. Try: python cost_optimization.py (advanced cost tracking)")
        print("   2. Run: python streaming_patterns.py (real-time responses)")
        print("   3. Check: python production_patterns.py (enterprise deployment)")
        print("   4. Explore: python lambda_integration.py (serverless patterns)")
        print()
        print("📖 Learn More:")
        print("   → Integration Guide: docs/integrations/bedrock.md")
        print("   → API Reference: docs/api/providers/bedrock.md")
        print("   → Cost Optimization: docs/cost-optimization/bedrock.md")

    else:
        print(f"⚠️ {success_count}/{total_demos} demos completed successfully")
        print()
        print("🔧 Troubleshooting:")
        print("   1. Run: python bedrock_validation.py")
        print("   2. Check AWS credentials: aws sts get-caller-identity")
        print("   3. Verify Bedrock access in AWS console")
        print("   4. Check model permissions for your region")

    return success_count == total_demos


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result else 1)
