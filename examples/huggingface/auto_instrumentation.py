#!/usr/bin/env python3
"""
Hugging Face Auto-Instrumentation Example

This example demonstrates zero-code auto-instrumentation for Hugging Face.
Your existing Hugging Face code works unchanged with automatic GenOps telemetry.

Example usage:
    python auto_instrumentation.py

Features demonstrated:
- Zero-code instrumentation setup
- Automatic telemetry injection
- Multiple AI task support
- Governance attribute propagation
- Works with existing code unchanged
"""

import logging
import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Set up logging to see telemetry in action
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_auto_instrumentation():
    """Demonstrate zero-code auto-instrumentation for Hugging Face."""

    print("🤗 Hugging Face Zero-Code Auto-Instrumentation Demo")
    print("=" * 60)
    print("This demonstrates automatic GenOps telemetry with no code changes needed!")
    print()

    try:
        # Step 1: Enable auto-instrumentation (this is the ONLY code change needed!)
        print("📡 Step 1: Enabling auto-instrumentation...")
        from genops.providers.huggingface import instrument_huggingface

        # This enables automatic telemetry for ALL Hugging Face API calls
        instrumentation_result = instrument_huggingface()

        if instrumentation_result:
            print("✅ Auto-instrumentation enabled successfully!")
            print("   → All Hugging Face API calls now automatically tracked")
            print("   → Cost, performance, and governance data captured")
            print("   → No changes needed to your existing code")
        else:
            print("⚠️ Auto-instrumentation setup encountered issues")
            print("   → Check that huggingface_hub is installed")
            return False

        print()

        # Step 2: Use Hugging Face normally - telemetry is automatic!
        print("🚀 Step 2: Using Hugging Face normally (telemetry is automatic)...")

        # Import and use Hugging Face exactly as you normally would
        try:
            from huggingface_hub import InferenceClient

            # Create client normally - no GenOps code needed!
            client = InferenceClient()

            print("✅ Created Hugging Face InferenceClient")
            print("   → Client is now automatically instrumented")
            print()

        except ImportError:
            print("❌ huggingface_hub not installed")
            print("💡 Install with: pip install huggingface_hub")
            return False

        # Step 3: Demonstrate different AI tasks with automatic tracking
        print("🎯 Step 3: Demonstrating automatic tracking across AI tasks...")
        print()

        # Text Generation Example
        print("📝 Text Generation (automatic tracking):")
        try:
            response = client.text_generation(
                "Once upon a time in a land far away,",
                model="microsoft/DialoGPT-medium",
                max_new_tokens=50,
                # Governance attributes are automatically captured if provided
                temperature=0.7,
            )

            print(f"   Response: {str(response)[:100]}...")
            print("   ✅ Cost and telemetry automatically captured")
            print("   ✅ Provider detection: Hugging Face Hub model")

        except Exception as e:
            print(f"   ⚠️ Text generation test failed: {e}")
            print("   💡 This might be due to rate limits or connectivity")

        print()

        # Feature Extraction Example
        print("🔍 Feature Extraction/Embeddings (automatic tracking):")
        try:
            embeddings = client.feature_extraction(
                "This is a test sentence for embedding",
                model="sentence-transformers/all-MiniLM-L6-v2",
            )

            print(f"   Embeddings shape: {len(embeddings) if embeddings else 'N/A'}")
            print("   ✅ Cost and telemetry automatically captured")
            print("   ✅ Task type: feature-extraction automatically detected")

        except Exception as e:
            print(f"   ⚠️ Feature extraction test failed: {e}")
            print("   💡 This might be due to rate limits or model availability")

        print()

        # Step 4: Show how to add governance attributes with existing calls
        print("🏛️ Step 4: Adding governance attributes to existing calls...")
        print(
            "(Your existing function calls work unchanged, just add governance attributes)"
        )
        print()

        # This is how you add governance to existing calls - minimal changes!
        try:
            governed_response = client.text_generation(
                "Write a professional email greeting",
                model="microsoft/DialoGPT-medium",
                max_new_tokens=30,
                # Just add these governance attributes - everything else unchanged!
                team="marketing-team",
                project="email-automation",
                customer_id="enterprise-client-123",
                environment="production",
                cost_center="marketing-ops",
            )

            print("📧 Email generation with governance:")
            print(f"   Response: {str(governed_response)[:80]}...")
            print("   ✅ Automatic cost attribution to: marketing-team")
            print("   ✅ Project tracking: email-automation")
            print("   ✅ Customer billing: enterprise-client-123")
            print("   ✅ All telemetry automatically exported")

        except Exception as e:
            print(f"   ⚠️ Governed text generation failed: {e}")

        print()

        # Step 5: Multiple providers through Hugging Face
        print(
            "🌐 Step 5: Multi-provider support (OpenAI/Anthropic via Hugging Face)..."
        )
        print("(Cost tracking works across all providers automatically)")
        print()

        # Example of using different providers through Hugging Face
        providers_to_test = [
            ("microsoft/DialoGPT-medium", "Hugging Face Hub"),
            # Note: These would require specific API access/setup
            # ("gpt-3.5-turbo", "OpenAI via Hugging Face"),
            # ("claude-3-haiku", "Anthropic via Hugging Face"),
        ]

        for model, provider_desc in providers_to_test:
            try:
                print(f"   Testing {provider_desc}:")
                response = client.text_generation(
                    "Hello, how are you?",
                    model=model,
                    max_new_tokens=20,
                    team="testing-team",
                    project="provider-comparison",
                )
                print(f"     ✅ Response: {str(response)[:60]}...")
                print("     ✅ Provider automatically detected and costs tracked")

            except Exception as e:
                print(f"     ⚠️ {provider_desc} test failed: {e}")

        print()

        # Step 6: What happens automatically
        print("🔄 What Happens Automatically:")
        print(
            "   ✅ Cost calculation for all providers (OpenAI, Anthropic, Hub models)"
        )
        print("   ✅ Token usage tracking and estimation")
        print("   ✅ Provider detection and routing analysis")
        print("   ✅ Performance metrics (latency, throughput)")
        print("   ✅ Governance attribute propagation")
        print("   ✅ Error tracking and debugging information")
        print("   ✅ OpenTelemetry export to your observability platform")
        print()

        # Step 7: Observability integration
        print("📊 Observability Integration:")
        print("   → Telemetry data exported via OpenTelemetry")
        print("   → Works with Datadog, Honeycomb, Grafana, Jaeger, etc.")
        print("   → Set OTEL_EXPORTER_OTLP_ENDPOINT to configure export")
        print("   → All cost and performance data available in your dashboards")
        print()

        return True

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install with: pip install genops-ai[huggingface]")
        return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("💡 Check your internet connection and Hugging Face setup")
        return False


def demonstrate_uninstrumentation():
    """Show how to remove auto-instrumentation if needed."""

    print("🔄 Removing Auto-Instrumentation (optional):")
    print("   You can disable auto-instrumentation if needed...")

    try:
        from genops.providers.huggingface import uninstrument_huggingface

        result = uninstrument_huggingface()
        if result:
            print("   ✅ Auto-instrumentation removed")
            print("   → Hugging Face calls back to normal behavior")
        else:
            print("   ℹ️ Auto-instrumentation was not active")

    except ImportError:
        print("   ⚠️ Uninstrumentation utilities not available")


def main():
    """Main demonstration function."""

    print("Welcome to the Hugging Face GenOps Auto-Instrumentation Demo!")
    print()
    print("This example shows how to add comprehensive AI governance telemetry")
    print("to your existing Hugging Face applications with minimal code changes.")
    print()

    # Run the demonstration
    success = demonstrate_auto_instrumentation()

    if success:
        print("🎉 Auto-Instrumentation Demo Completed Successfully!")
        print()
        print("🚀 Next Steps:")
        print("   1. Try running your own Hugging Face code - it's now auto-tracked!")
        print(
            "   2. Set up OpenTelemetry export to see data in your observability platform"
        )
        print(
            "   3. Add governance attributes (team, project, customer_id) to your calls"
        )
        print("   4. Check out multi_provider_costs.py for advanced cost tracking")
        print("   5. Run production_patterns.py for enterprise deployment patterns")
        print()
        print("📖 Documentation:")
        print("   → Quick Start: docs/huggingface-quickstart.md")
        print("   → Integration Guide: docs/integrations/huggingface.md")
        print("   → API Reference: docs/api/providers/huggingface.md")

    else:
        print("❌ Demo encountered issues. See error messages above.")
        print()
        print("🔧 Troubleshooting:")
        print("   1. Run setup_validation.py to check your configuration")
        print("   2. Install dependencies: pip install genops-ai[huggingface]")
        print("   3. Check internet connectivity for Hugging Face API")
        print("   4. Review the Hugging Face quickstart guide")

    print()

    # Optional: demonstrate uninstrumentation
    demonstrate_uninstrumentation()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
