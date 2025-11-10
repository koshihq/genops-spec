#!/usr/bin/env python3
"""
Bedrock Auto-Instrumentation Example

This example demonstrates zero-code auto-instrumentation for AWS Bedrock.
Works with existing Bedrock applications unchanged, adding comprehensive
governance and cost intelligence automatically.

Example usage:
    python auto_instrumentation.py

Features demonstrated:
- Zero-code instrumentation for existing Bedrock applications
- Automatic telemetry injection for all Bedrock API calls
- Multi-model support with automatic provider detection
- Governance attribute propagation
- Real-time cost tracking across different models
"""

import json
import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def demonstrate_auto_instrumentation():
    """Demonstrate auto-instrumentation with various Bedrock models."""

    print("🔧 GenOps Bedrock Auto-Instrumentation Demo")
    print("=" * 50)
    print("This shows how GenOps adds governance to existing Bedrock code")
    print("without requiring any code changes to your application.")
    print()

    try:
        # Step 1: Enable auto-instrumentation (this is the ONLY line you need!)
        print("📡 Enabling GenOps auto-instrumentation...")
        from genops.providers.bedrock import instrument_bedrock
        instrument_bedrock()
        print("✅ Auto-instrumentation enabled! All Bedrock calls now tracked.")
        print()

        # Step 2: Use existing Bedrock code unchanged
        print("🏗️  Your existing Bedrock code works exactly the same...")
        import boto3

        # This is your normal, unchanged Bedrock code
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

        # Example 1: Claude text generation (unchanged existing code)
        print("\n📝 Testing Claude 3 Haiku (your existing code):")
        claude_response = bedrock_runtime.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "prompt": "\n\nHuman: Explain quantum computing in one sentence.\n\nAssistant:",
                "max_tokens_to_sample": 50,
                "temperature": 0.7
            }),
            contentType="application/json",
            accept="application/json"
        )

        claude_result = json.loads(claude_response['body'].read())
        print(f"   🤖 Response: {claude_result.get('completion', 'No response').strip()}")
        print("   ✅ Automatically tracked: cost, latency, governance")

        # Example 2: Amazon Titan (unchanged existing code)
        print("\n📝 Testing Amazon Titan Text Express:")
        try:
            titan_response = bedrock_runtime.invoke_model(
                modelId="amazon.titan-text-express-v1",
                body=json.dumps({
                    "inputText": "What is machine learning?",
                    "textGenerationConfig": {
                        "maxTokenCount": 50,
                        "temperature": 0.7
                    }
                }),
                contentType="application/json",
                accept="application/json"
            )

            titan_result = json.loads(titan_response['body'].read())
            titan_text = titan_result.get('results', [{}])[0].get('outputText', 'No response')
            print(f"   🤖 Response: {titan_text.strip()}")
            print("   ✅ Automatically tracked: different model, same governance")

        except Exception as e:
            print(f"   ⚠️  Titan not available: {str(e)[:60]}...")
            print("   💡 Some models need to be enabled in AWS console")

        # Example 3: AI21 Jurassic (if available)
        print("\n📝 Testing AI21 Jurassic-2 Mid:")
        try:
            j2_response = bedrock_runtime.invoke_model(
                modelId="ai21.j2-mid-v1",
                body=json.dumps({
                    "prompt": "The future of artificial intelligence is",
                    "maxTokens": 30,
                    "temperature": 0.8
                }),
                contentType="application/json",
                accept="application/json"
            )

            j2_result = json.loads(j2_response['body'].read())
            j2_text = j2_result.get('completions', [{}])[0].get('data', {}).get('text', 'No response')
            print(f"   🤖 Response: {j2_text.strip()}")
            print("   ✅ Automatically tracked: multi-provider cost comparison")

        except Exception as e:
            print(f"   ⚠️  Jurassic not available: {str(e)[:60]}...")

        print()
        print("🎉 Amazing! All of your existing Bedrock code now has:")
        print("   💰 Automatic cost calculation (per model, per region)")
        print("   🏷️  Automatic provider detection (Anthropic, Amazon, AI21, etc.)")
        print("   📊 Performance metrics (latency, tokens, success rates)")
        print("   🔍 Error tracking and debugging context")
        print("   📡 OpenTelemetry export to your observability stack")
        print("   🏛️  Enterprise governance (when you add attributes)")
        print()
        print("💡 Pro tip: Add governance attributes to your calls:")
        print('   # Just add these parameters to your existing invoke_model calls')
        print('   team="ai-team", project="chatbot", customer_id="enterprise-123"')

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Install GenOps with Bedrock support:")
        print("   pip install genops-ai[bedrock]")
        return False

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        print("\n💡 Common solutions:")
        print("   - Run: python bedrock_validation.py")
        print("   - Check AWS credentials and region configuration")
        print("   - Enable model access in AWS Bedrock console")
        return False


def demonstrate_streaming():
    """Demonstrate auto-instrumentation with streaming responses."""

    print("\n🌊 Streaming Response Auto-Instrumentation")
    print("-" * 45)

    try:
        import boto3
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

        print("📡 Testing streaming with Claude (auto-instrumented)...")

        # Streaming is also automatically tracked!
        response = bedrock_runtime.invoke_model_with_response_stream(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "prompt": "\n\nHuman: Write a haiku about AI.\n\nAssistant:",
                "max_tokens_to_sample": 100,
                "temperature": 0.8
            }),
            contentType="application/json",
            accept="application/json"
        )

        print("   🤖 Streaming response: ", end="", flush=True)
        full_response = ""

        for event in response['body']:
            if 'chunk' in event:
                chunk_data = json.loads(event['chunk']['bytes'])
                chunk_text = chunk_data.get('completion', '')
                if chunk_text:
                    print(chunk_text, end="", flush=True)
                    full_response += chunk_text

        print(f"\n   ✅ Streaming also auto-tracked: {len(full_response)} characters generated")
        print("   📊 Telemetry includes: streaming latency, chunk count, total cost")

    except Exception as e:
        print(f"   ⚠️  Streaming demo failed: {str(e)[:60]}...")
        print("   💡 Streaming may not be available for all models")


def show_governance_enhancement():
    """Show how to add governance attributes to existing code."""

    print("\n🏛️  Adding Governance to Existing Code")
    print("-" * 40)
    print("Your existing code can be enhanced with just environment variables:")
    print()

    # Show environment variable setup
    env_vars = {
        "GENOPS_DEFAULT_TEAM": "ai-engineering",
        "GENOPS_DEFAULT_PROJECT": "customer-chatbot",
        "GENOPS_DEFAULT_ENVIRONMENT": "production",
        "GENOPS_DEFAULT_COST_CENTER": "AI-Platform"
    }

    print("💡 Set these environment variables for automatic governance:")
    for var, value in env_vars.items():
        print(f"   export {var}='{value}'")

    print()
    print("🎯 Or add governance directly in code (no API changes needed):")
    print("   # GenOps detects and uses these attributes automatically")
    print("   # Your existing invoke_model calls get enhanced governance")
    print("   # Customer attribution, cost centers, compliance tracking")


def main():
    """Main demonstration function."""

    success = demonstrate_auto_instrumentation()

    if success:
        demonstrate_streaming()
        show_governance_enhancement()

        print("\n✅ Auto-instrumentation Demo Complete!")
        print()
        print("🚀 Key Takeaways:")
        print("   1. Zero code changes needed - just call instrument_bedrock()")
        print("   2. All existing Bedrock calls automatically get governance")
        print("   3. Multi-model support with automatic provider detection")
        print("   4. Streaming and batch operations both supported")
        print("   5. Add governance with environment variables or attributes")
        print()
        print("🎯 Next Steps:")
        print("   → Try: python basic_tracking.py (manual adapter control)")
        print("   → Advanced: python cost_optimization.py (cost intelligence)")
        print("   → Production: python production_patterns.py (enterprise features)")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
