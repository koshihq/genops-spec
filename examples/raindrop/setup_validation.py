#!/usr/bin/env python3
"""
Raindrop AI + GenOps Setup Validation

This script validates the complete setup for Raindrop AI integration with GenOps,
including environment configuration, authentication, and system requirements.

Usage:
    python setup_validation.py

Environment Variables:
    RAINDROP_API_KEY: Your Raindrop AI API key
    GENOPS_TEAM: Team identifier for cost attribution (optional)
    GENOPS_PROJECT: Project identifier for cost attribution (optional)
    GENOPS_ENVIRONMENT: Environment (development/staging/production)
    GENOPS_DAILY_BUDGET_LIMIT: Daily budget limit in USD (optional)

Example:
    export RAINDROP_API_KEY="your-raindrop-api-key"
    export GENOPS_TEAM="ai-platform"
    export GENOPS_PROJECT="agent-monitoring"
    python setup_validation.py

Author: GenOps AI Contributors
"""

import os
import sys
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from genops.providers.raindrop_validation import validate_setup, print_validation_result, validate_setup_interactive
except ImportError as e:
    print(f"❌ Error importing GenOps Raindrop validation: {e}")
    print("💡 Make sure you're in the project root directory and GenOps is properly installed")
    sys.exit(1)

def main():
    """Main validation workflow."""
    print("🔍 Raindrop AI + GenOps Setup Validation")
    print("=" * 60)
    
    # Check if this is an interactive session
    interactive = len(sys.argv) > 1 and sys.argv[1] == "--interactive"
    
    if interactive:
        print("🔧 Running in interactive mode...")
        result = validate_setup_interactive()
    else:
        # Check basic environment configuration first
        print("\n📋 Environment Configuration Check:")
        
        api_key = os.getenv("RAINDROP_API_KEY")
        team = os.getenv("GENOPS_TEAM")
        project = os.getenv("GENOPS_PROJECT")
        environment = os.getenv("GENOPS_ENVIRONMENT", "production")
        budget_limit = os.getenv("GENOPS_DAILY_BUDGET_LIMIT")
        
        print(f"  {'✅' if api_key else '❌'} RAINDROP_API_KEY {'configured' if api_key else 'not found'}")
        print(f"  {'✅' if team else '⚠️'} GENOPS_TEAM {'configured' if team else 'not set (will use default)'}")
        print(f"  {'✅' if project else '⚠️'} GENOPS_PROJECT {'configured' if project else 'not set (will use default)'}")
        print(f"  ℹ️  GENOPS_ENVIRONMENT: {environment}")
        if budget_limit:
            print(f"  ✅ GENOPS_DAILY_BUDGET_LIMIT: ${budget_limit}")
        
        # Run comprehensive validation
        result = validate_setup(api_key)
    
    # Display detailed results
    print_validation_result(result, verbose=True)
    
    # Provide next steps guidance
    if result.is_valid:
        print("🚀 Setup validation completed successfully!")
        print("\n📚 Next Steps:")
        print("  1. Try basic tracking: python basic_tracking.py")
        print("  2. Explore auto-instrumentation: python auto_instrumentation.py")
        print("  3. Check advanced features: python advanced_features.py")
        print("  4. Review cost optimization: python cost_optimization.py")
    else:
        print("❌ Setup validation failed!")
        print("\n🔧 Troubleshooting:")
        print("  1. Fix the error-level issues listed above")
        print("  2. Check the integration guide: docs/integrations/raindrop.md")
        print("  3. Run interactive setup: python setup_validation.py --interactive")
        print("  4. Get help: https://github.com/KoshiHQ/GenOps-AI/discussions")
        
        # Exit with error code for CI/CD integration
        sys.exit(1)

if __name__ == "__main__":
    main()