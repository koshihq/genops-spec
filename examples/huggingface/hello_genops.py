#!/usr/bin/env python3
"""
Ultra-Simple GenOps Hello World Example

This is the simplest possible example to verify GenOps is working.
Perfect for first-time users to confirm everything is set up correctly.

Example usage:
    python hello_genops.py

What this demonstrates:
- Zero-code instrumentation setup
- Basic AI operation with automatic governance
- Immediate confirmation that GenOps is working
"""

import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def main():
    """The simplest possible GenOps example."""
    
    print("👋 GenOps Hello World Example")
    print("=" * 35)
    print("This is the simplest way to confirm GenOps is working.")
    print()
    
    try:
        # Step 1: Enable GenOps instrumentation
        print("📡 Enabling GenOps instrumentation...")
        from genops.providers.huggingface import instrument_huggingface
        instrument_huggingface()
        print("✅ GenOps instrumentation enabled!")
        
        # Step 2: Use Hugging Face normally
        print("\n🤗 Making Hugging Face API call...")
        from huggingface_hub import InferenceClient
        
        client = InferenceClient()
        
        # This single line now has comprehensive AI governance!
        result = client.text_generation(
            "Hello GenOps!", 
            model="microsoft/DialoGPT-medium",
            max_new_tokens=20
        )
        
        # Step 3: Celebrate success!
        print("✅ Success! AI operation completed with GenOps governance!")
        print(f"🤖 AI Response: {result}")
        print()
        print("🎉 Congratulations! GenOps is now tracking:")
        print("   💰 Cost calculation and attribution")
        print("   🏛️  Governance and compliance data") 
        print("   📊 Performance and usage metrics")
        print("   🔍 Error tracking and debugging info")
        print("   📡 OpenTelemetry export to your observability platform")
        print()
        print("🚀 You're ready to explore more advanced GenOps features!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Fix this by installing GenOps with Hugging Face support:")
        print("   pip install genops-ai[huggingface]")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   Error type: {type(e).__name__}")
        print("\n💡 This might help:")
        print("   - Check your internet connection")
        print("   - Verify Hugging Face Hub is accessible")
        print("   - Try a different model if this one is unavailable")
        print("   - Run the validation script: python setup_validation.py")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎯 What's Next?")
        print("   1. Try: python basic_tracking.py")
        print("   2. Explore: python cost_tracking.py")  
        print("   3. Advanced: python huggingface_specific_advanced.py")
        print("   4. Production: python production_patterns.py")
        
    sys.exit(0 if success else 1)