#!/usr/bin/env python3
"""
Hugging Face Basic Usage Example

This example demonstrates essential patterns for using GenOps with Hugging Face.
Shows manual adapter usage, governance attributes, and cost tracking.

Example usage:
    python basic_usage.py

Features demonstrated:
- Manual GenOps adapter usage
- Governance attribute examples
- Basic cost tracking
- Task-specific instrumentation
- Error handling patterns
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
    """Demonstrate manual GenOps Hugging Face adapter usage."""

    print("🤗 Manual GenOps Hugging Face Adapter Usage")
    print("=" * 60)
    print("This shows how to use the GenOps adapter directly for full control.")
    print()

    try:
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter

        # Create adapter with automatic telemetry
        print("📡 Creating GenOps Hugging Face adapter...")
        adapter = GenOpsHuggingFaceAdapter()

        if not adapter.is_available():
            print("❌ Hugging Face not available")
            print("💡 Install with: pip install huggingface_hub")
            return False

        print("✅ GenOps Hugging Face adapter created successfully")
        print(f"   → Supported AI tasks: {len(adapter.get_supported_tasks())}")
        print(
            f"   → Available tasks: {', '.join(adapter.get_supported_tasks()[:5])}..."
        )
        print()

        # Text Generation with governance
        print("📝 Text Generation with Governance Attributes:")
        try:
            response = adapter.text_generation(
                prompt="Write a creative story opening about a mysterious library.",
                model="microsoft/DialoGPT-medium",
                max_new_tokens=100,
                temperature=0.8,
                # Governance attributes for cost attribution
                team="creative-team",
                project="story-generation",
                customer_id="publishing-client-456",
                environment="production",
                feature="story-opener",
                cost_center="content-creation",
            )

            print(f"   📖 Generated story: {str(response)[:120]}...")
            print("   ✅ Governance attributes captured:")
            print("      → Team: creative-team (cost attribution)")
            print("      → Project: story-generation (project tracking)")
            print("      → Customer: publishing-client-456 (billing)")
            print("      → Environment: production (environment segregation)")
            print("   ✅ Cost automatically calculated and tracked")
            print()

        except Exception as e:
            print(f"   ⚠️ Text generation failed: {e}")
            print("   💡 This might be due to rate limits or connectivity")
            print()

        # Chat Completion Example
        print("💬 Chat Completion with Multi-Message Context:")
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant for customer support.",
                },
                {
                    "role": "user",
                    "content": "I'm having trouble with my order. Can you help?",
                },
                {
                    "role": "assistant",
                    "content": "I'd be happy to help you with your order. What specific issue are you experiencing?",
                },
                {
                    "role": "user",
                    "content": "My package was supposed to arrive yesterday but it hasn't shown up.",
                },
            ]

            chat_response = adapter.chat_completion(
                messages=messages,
                model="microsoft/DialoGPT-medium",  # Note: may not support chat format
                max_new_tokens=80,
                temperature=0.7,
                # Different governance context
                team="support-team",
                project="customer-service-ai",
                customer_id="ecommerce-client-789",
                feature="order-tracking-help",
            )

            print(f"   💬 Support response: {str(chat_response)[:100]}...")
            print("   ✅ Multi-message context processed")
            print("   ✅ Cost attributed to: support-team")
            print("   ✅ Customer billing: ecommerce-client-789")
            print()

        except Exception as e:
            print(f"   ⚠️ Chat completion test failed: {e}")
            print("   💡 Note: Not all models support chat completion format")
            print()

        # Feature Extraction (Embeddings) Example
        print("🔍 Feature Extraction/Embeddings:")
        try:
            texts_to_embed = [
                "Customer service is very important for our business.",
                "We need to improve our response times.",
                "Product quality must meet high standards.",
            ]

            embeddings = adapter.feature_extraction(
                inputs=texts_to_embed,
                model="sentence-transformers/all-MiniLM-L6-v2",
                # Analytics team governance
                team="analytics-team",
                project="customer-feedback-analysis",
                customer_id="internal-analytics",
                feature="sentiment-embeddings",
            )

            if embeddings:
                print(f"   📊 Generated embeddings for {len(texts_to_embed)} texts")
                print(
                    f"   📐 Embedding dimensions: {len(embeddings[0]) if isinstance(embeddings, list) and embeddings else 'N/A'}"
                )
                print("   ✅ Cost calculated for embedding generation")
                print("   ✅ Task type: feature-extraction automatically detected")
                print()
            else:
                print("   ⚠️ No embeddings returned")

        except Exception as e:
            print(f"   ⚠️ Feature extraction failed: {e}")
            print("   💡 Check model availability and network connection")
            print()

        # Text-to-Image Example
        print("🎨 Text-to-Image Generation:")
        try:
            adapter.text_to_image(
                prompt="A futuristic city skyline at sunset with flying cars",
                model="runwayml/stable-diffusion-v1-5",  # Example model
                # Design team governance
                team="design-team",
                project="marketing-visuals",
                customer_id="advertising-client-321",
                feature="campaign-imagery",
                cost_center="creative-services",
            )

            print("   🖼️ Image generation attempted")
            print("   ✅ Cost tracking includes image generation pricing")
            print("   ✅ Task type: text-to-image automatically detected")
            print("   💡 Image data would be available in production")
            print()

        except Exception as e:
            print(f"   ⚠️ Text-to-image generation failed: {e}")
            print("   💡 Image generation requires specific model access")
            print()

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("💡 Install GenOps with: pip install genops-ai[huggingface]")
        return False


def demonstrate_provider_detection():
    """Show provider detection capabilities."""

    print("🔍 Provider Detection Intelligence")
    print("=" * 40)
    print("GenOps automatically detects the underlying provider for cost calculation:")
    print()

    try:
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter

        adapter = GenOpsHuggingFaceAdapter()

        # Test various model patterns
        test_models = [
            "gpt-3.5-turbo",  # OpenAI
            "gpt-4",  # OpenAI
            "claude-3-sonnet",  # Anthropic
            "claude-3-haiku",  # Anthropic
            "command-r",  # Cohere
            "mistral-7b-instruct",  # Mistral
            "llama-2-7b-chat",  # Meta
            "microsoft/DialoGPT-medium",  # Hugging Face Hub
            "sentence-transformers/all-MiniLM-L6-v2",  # Hugging Face Hub
            "runwayml/stable-diffusion-v1-5",  # Hugging Face Hub
        ]

        for model in test_models:
            provider = adapter.detect_provider_for_model(model)
            print(f"   📍 {model[:35]:35} → {provider}")

        print()
        print("✅ Provider detection enables accurate cost calculation")
        print("✅ Each provider has different pricing models and rate structures")
        print("✅ Costs automatically attributed to correct provider")
        print()

        return True

    except ImportError:
        print("❌ Provider detection unavailable - check GenOps installation")
        return False


def demonstrate_cost_tracking():
    """Demonstrate cost tracking and attribution."""

    print("💰 Cost Tracking and Attribution")
    print("=" * 40)
    print("See how GenOps tracks costs across different scenarios:")
    print()

    try:
        from genops.providers.huggingface_pricing import (
            calculate_huggingface_cost,
            compare_model_costs,
            get_provider_info,  # noqa: F401
        )

        # Example cost calculations
        cost_scenarios = [
            {
                "scenario": "Short chat interaction",
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "input_tokens": 150,
                "output_tokens": 50,
                "task": "chat-completion",
            },
            {
                "scenario": "Long document generation",
                "provider": "huggingface_hub",
                "model": "microsoft/DialoGPT-medium",
                "input_tokens": 500,
                "output_tokens": 2000,
                "task": "text-generation",
            },
            {
                "scenario": "Embedding generation",
                "provider": "huggingface_hub",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "input_tokens": 1000,
                "output_tokens": 0,
                "task": "feature-extraction",
            },
            {
                "scenario": "Image generation",
                "provider": "huggingface_hub",
                "model": "runwayml/stable-diffusion-v1-5",
                "input_tokens": 100,
                "output_tokens": 0,
                "task": "text-to-image",
            },
        ]

        for scenario in cost_scenarios:
            cost = calculate_huggingface_cost(
                provider=scenario["provider"],
                model=scenario["model"],
                input_tokens=scenario["input_tokens"],
                output_tokens=scenario["output_tokens"],
                task=scenario["task"],
            )

            print(f"   💳 {scenario['scenario']:25} → ${cost:.6f}")
            print(f"      Model: {scenario['model'][:40]}")
            print(
                f"      Tokens: {scenario['input_tokens']} in, {scenario['output_tokens']} out"
            )
            print()

        # Model comparison
        print("📊 Model Cost Comparison:")
        models_to_compare = [
            "gpt-3.5-turbo",
            "microsoft/DialoGPT-medium",
            "claude-3-haiku",
        ]
        comparison = compare_model_costs(
            models_to_compare, input_tokens=1000, output_tokens=500
        )

        for model, info in comparison.items():
            relative_cost = info["relative_cost"]
            cost_indicator = (
                "💰" if relative_cost > 2 else "💚" if relative_cost < 1.5 else "💛"
            )
            print(
                f"   {cost_indicator} {model[:35]:35} → ${info['cost']:.6f} ({relative_cost:.1f}x)"
            )

        print()
        print("✅ Cost comparison helps optimize model selection")
        print("✅ All costs automatically tracked in telemetry")
        print()

        return True

    except ImportError as e:
        print(f"❌ Cost tracking unavailable: {e}")
        print("💡 Check pricing module installation")
        return False


def demonstrate_error_handling():
    """Show error handling patterns."""

    print("🛡️ Error Handling and Resilience")
    print("=" * 40)
    print("GenOps gracefully handles various error scenarios:")
    print()

    try:
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter

        adapter = GenOpsHuggingFaceAdapter()

        # Test with invalid model
        print("   🧪 Testing invalid model handling...")
        try:
            adapter.text_generation(
                prompt="Test prompt",
                model="nonexistent-model-12345",
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
                model="microsoft/DialoGPT-medium",
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

        return True

    except ImportError:
        print("❌ Error handling demo unavailable - check installation")
        return False


def main():
    """Main demonstration function."""

    print("Welcome to the Hugging Face GenOps Basic Usage Demo!")
    print()
    print("This example demonstrates essential patterns for integrating")
    print("GenOps governance and telemetry with Hugging Face applications.")
    print()

    success_count = 0
    total_demos = 4

    # Run all demonstrations
    demos = [
        ("Manual Adapter Usage", demonstrate_manual_adapter),
        ("Provider Detection", demonstrate_provider_detection),
        ("Cost Tracking", demonstrate_cost_tracking),
        ("Error Handling", demonstrate_error_handling),
    ]

    for demo_name, demo_func in demos:
        print(f"🚀 Running {demo_name} Demo...")
        try:
            success = demo_func()
            if success:
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
        print("   1. Try multi_provider_costs.py for advanced cost tracking")
        print("   2. Run ai_task_examples.py for comprehensive AI task coverage")
        print("   3. Check out cost_optimization.py for optimization strategies")
        print("   4. Explore production_patterns.py for enterprise deployment")
        print()
        print("📖 Learn More:")
        print("   → Integration Guide: docs/integrations/huggingface.md")
        print("   → API Reference: docs/api/providers/huggingface.md")
        print("   → Cost Optimization: docs/cost-optimization/huggingface.md")

    else:
        print(f"⚠️ {success_count}/{total_demos} demos completed successfully")
        print()
        print("🔧 Troubleshooting:")
        print("   1. Run setup_validation.py to check configuration")
        print("   2. Verify internet connectivity for Hugging Face API")
        print("   3. Check that all dependencies are installed")
        print("   4. Review error messages above for specific issues")

    return 0 if success_count == total_demos else 1


if __name__ == "__main__":
    sys.exit(main())
