#!/usr/bin/env python3
"""
Ultra-Simple GenOps Bedrock Hello World Example

This is the simplest possible example to verify GenOps Bedrock integration is working.
Perfect for first-time users to confirm everything is set up correctly.

Example usage:
    python hello_genops.py

What this demonstrates:
- Zero-code instrumentation setup with Bedrock
- Basic AI operation with automatic governance
- Immediate confirmation that GenOps is working with AWS Bedrock
- AWS credential validation and region setup
"""

"""
Note: This example assumes genops-ai is installed via pip.
For development, install in editable mode: pip install -e .
"""

def main():
    """The simplest possible GenOps Bedrock example."""

    print("👋 GenOps Bedrock Hello World Example")
    print("=" * 40)
    print("This is the simplest way to confirm GenOps Bedrock is working.")
    print()

    try:
        # Step 1: Enable GenOps instrumentation for Bedrock
        print("📡 Enabling GenOps Bedrock instrumentation...")
        from genops.providers.bedrock import instrument_bedrock
        instrument_bedrock()
        print("✅ GenOps Bedrock instrumentation enabled!")

        # Step 2: Use Bedrock normally with boto3
        print("\n🏗️  Making AWS Bedrock API call...")
        import json

        import boto3

        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

        # This single call now has comprehensive AI governance!
        response = bedrock_runtime.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": "Hello GenOps!"}],
                "max_tokens": 20,
                "anthropic_version": "bedrock-2023-05-31"
            }),
            contentType="application/json"
        )

        # Extract and display response
        response_body = json.loads(response['body'].read())
        ai_response = response_body.get('content', [{}])[0].get('text', 'Hello from Claude!')

        # Step 3: Celebrate success!
        print("✅ Success! AI operation completed with GenOps governance!")
        print(f"🤖 Claude Response: {ai_response.strip()}")
        print()
        print("🎉 Congratulations! GenOps is now tracking:")
        print("   💰 Cost calculation and attribution across AWS regions")
        print("   🏛️  Governance and compliance data with CloudTrail integration")
        print("   📊 Performance and usage metrics with AWS Cost Explorer")
        print("   🔍 Error tracking and debugging info")
        print("   📡 OpenTelemetry export to your observability platform")
        print()
        print("🚀 You're ready to explore more advanced GenOps Bedrock features!")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Fix this by installing GenOps with Bedrock support:")
        print("   pip install genops-ai[bedrock]")
        print("   # or")
        print("   pip install genops-ai boto3")
        return False

    except Exception as e:
        error_str = str(e)
        print(f"❌ Error: {error_str}")
        print(f"   Error type: {type(e).__name__}")
        print("\n💡 Common fixes:")

        if "credentials" in error_str.lower() or "NoCredentialsError" in str(type(e)):
            print("   - Configure AWS credentials: aws configure")
            print("   - Or set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
            print("   - Or use IAM roles if running on AWS infrastructure")
        elif "region" in error_str.lower():
            print("   - Verify Bedrock is available in your region (try us-east-1)")
            print("   - Set AWS_DEFAULT_REGION environment variable")
        elif "AccessDeniedException" in error_str:
            print("   - Enable model access in AWS Bedrock console")
            print("   - Add bedrock:* permissions to your IAM policy")
        elif "ValidationException" in error_str:
            print("   - Model may not be available in your region")
            print("   - Try a different model or region")
        else:
            print("   - Check your internet connection")
            print("   - Verify AWS Bedrock service is accessible")
            print("   - Run validation script: python bedrock_validation.py")
            print("   - Check AWS service status")

        return False

if __name__ == "__main__":
    success = main()

    if success:
        print("\n🎯 What's Next?")
        print("   1. Try: python auto_instrumentation.py")
        print("   2. Explore: python basic_tracking.py")
        print("   3. Advanced: python cost_optimization.py")
        print("   4. Production: python production_patterns.py")
        print("\n📖 Learn More:")
        print("   → Quickstart: docs/bedrock-quickstart.md")
        print("   → Full Guide: docs/integrations/bedrock.md")

    sys.exit(0 if success else 1)
