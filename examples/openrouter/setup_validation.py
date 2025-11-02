#!/usr/bin/env python3
"""
OpenRouter Setup Validation Example

This script validates that your OpenRouter + GenOps integration is properly configured
and working correctly. It checks environment variables, dependencies, connectivity,
and basic functionality.

Usage:
    python setup_validation.py
    
Expected output:
    ✅ Overall Status: VALID
    📊 Summary: X issues found
    💡 Recommendations: Setup looks good!
"""

import sys
import os

def main():
    """Run comprehensive OpenRouter setup validation."""
    print("🚀 GenOps + OpenRouter Setup Validation")
    print("=" * 50)
    
    try:
        # Import validation utilities
        from genops.providers.openrouter import validate_setup, print_validation_result
        
        print("🔍 Running comprehensive setup validation...")
        print("   • Checking environment variables")
        print("   • Validating dependencies") 
        print("   • Testing OpenRouter connectivity")
        print("   • Verifying GenOps configuration")
        print("   • Testing basic functionality")
        print()
        
        # Run validation
        result = validate_setup()
        
        # Display results in user-friendly format
        print_validation_result(result)
        
        # Exit with appropriate code
        if result.is_valid:
            print("🎉 Validation completed successfully!")
            sys.exit(0)
        else:
            print("⚠️  Please fix the issues above and re-run validation.")
            sys.exit(1)
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print()
        print("💡 Quick fixes:")
        print("   • Install GenOps: pip install genops-ai")
        print("   • Install OpenAI (for OpenRouter): pip install openai")
        print("   • Ensure you're in the correct Python environment")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Validation Error: {e}")
        print()
        print("💡 This might indicate a setup issue. Please check:")
        print("   • Environment variables (OPENROUTER_API_KEY)")
        print("   • Network connectivity")
        print("   • Package installations")
        sys.exit(1)


def quick_setup_guide():
    """Display quick setup guide for first-time users."""
    print("\n📚 Quick Setup Guide")
    print("-" * 30)
    print("1. Get OpenRouter API key:")
    print("   → Visit https://openrouter.ai/keys")
    print("   → Create account and generate API key")
    print()
    print("2. Set environment variable:")
    print("   export OPENROUTER_API_KEY='your-key-here'")
    print()
    print("3. Install dependencies:")
    print("   pip install genops-ai openai")
    print()
    print("4. Run validation:")
    print("   python setup_validation.py")
    print()


if __name__ == "__main__":
    # Check if this looks like a first-time setup
    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("👋 Welcome to GenOps + OpenRouter!")
        print("It looks like this might be your first time setting up.")
        quick_setup_guide()
        
        response = input("Continue with validation anyway? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Come back after setup! 👍")
            sys.exit(0)
        print()
    
    main()