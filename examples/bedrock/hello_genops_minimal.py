#!/usr/bin/env python3
"""
Minimal GenOps Bedrock Example

This is the absolute simplest way to verify GenOps Bedrock integration works.
Perfect for first-time users - just run it!

Usage:
    python hello_genops_minimal.py
"""

def main():
    print("🚀 Testing GenOps with AWS Bedrock...")
    
    try:
        # Step 1: Enable GenOps tracking
        from genops.providers.bedrock import auto_instrument_bedrock
        auto_instrument_bedrock()
        print("✅ GenOps auto-instrumentation enabled")
        
        # Step 2: Use Bedrock normally - now with GenOps tracking!
        import boto3
        import json
        
        client = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        response = client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": "Say hello!"}],
                "max_tokens": 20,
                "anthropic_version": "bedrock-2023-05-31"
            }),
            contentType="application/json"
        )
        
        print("✅ Success! GenOps is now tracking your Bedrock usage!")
        print("💰 Cost tracking, team attribution, and governance are active.")
        
        return True
        
    except ImportError:
        print("❌ GenOps not installed. Run: pip install genops-ai[bedrock]")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try running the validation: python -c \"from genops.providers.bedrock import validate_setup, print_validation_result; print_validation_result(validate_setup())\"")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎯 Next Steps:")
        print("   • Try: python auto_instrumentation.py")
        print("   • Learn: python basic_tracking.py")
        print("   • Advanced: python cost_optimization.py")
    
    exit(0 if success else 1)