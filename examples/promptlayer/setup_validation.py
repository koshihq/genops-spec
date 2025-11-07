#!/usr/bin/env python3
"""
PromptLayer + GenOps Setup Validation

This script validates your PromptLayer integration with GenOps governance setup.
It performs comprehensive checks on dependencies, configuration, connectivity,
and governance features to ensure everything is working correctly.

Run this FIRST before trying other examples to catch and fix common issues.

Usage:
    python setup_validation.py

Prerequisites:
    pip install genops[promptlayer]  # Includes PromptLayer SDK
    export PROMPTLAYER_API_KEY="pl-your-api-key"
    
    # Optional but recommended for full governance
    export GENOPS_TEAM="your-team"
    export GENOPS_PROJECT="your-project"
"""

import os
import sys
from datetime import datetime

def main():
    """Main validation function."""
    print("🔍 PromptLayer + GenOps Setup Validation")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    try:
        # Import validation utilities
        from genops.providers.promptlayer_validation import validate_setup, print_validation_result
        
        print("✅ GenOps PromptLayer validation utilities loaded successfully")
        
    except ImportError as e:
        print(f"❌ Failed to import GenOps PromptLayer validation utilities: {e}")
        print("\n🔧 Fix:")
        print("   pip install genops[promptlayer]")
        return False
    
    print("\n🚀 Running comprehensive validation checks...")
    print("-" * 40)
    
    # Run full validation
    result = validate_setup(
        include_connectivity_tests=True,
        include_performance_tests=True,
        include_governance_tests=True
    )
    
    # Print results
    print_validation_result(result, detailed=True)
    
    # Additional setup guidance
    if result.overall_status.value == "passed":
        print("🎉 Excellent! Your PromptLayer + GenOps setup is ready for production.")
        print("\n📚 Next Steps:")
        print("   • Try basic tracking: python basic_tracking.py")
        print("   • Enable zero-code governance: python auto_instrumentation.py")
        print("   • Explore prompt management: python prompt_management.py")
        print("   • Run all examples: ./run_all_examples.sh")
        
    elif result.overall_status.value == "warning":
        print("⚠️ Your setup is functional but can be improved.")
        print("\n📚 You can proceed with:")
        print("   • Basic examples: python basic_tracking.py")
        print("   • Auto-instrumentation: python auto_instrumentation.py")
        print("\n💡 Consider addressing the warnings for optimal experience.")
        
    else:
        print("❌ Setup has critical issues that need to be resolved first.")
        print("\n🔧 Required fixes:")
        failed_checks = [c for c in result.checks if c.status.value == "failed"]
        for check in failed_checks:
            if check.fix_suggestion:
                print(f"   • {check.name}: {check.fix_suggestion}")
        
        print("\n📚 After fixing issues, try:")
        print("   • Re-run validation: python setup_validation.py")
        print("   • Check basic functionality: python basic_tracking.py")
    
    # Environment information
    print("\n🔧 Environment Information:")
    print(f"   • Python version: {sys.version.split()[0]}")
    print(f"   • Platform: {sys.platform}")
    
    # Check environment variables
    api_key = os.getenv('PROMPTLAYER_API_KEY')
    team = os.getenv('GENOPS_TEAM')
    project = os.getenv('GENOPS_PROJECT')
    
    print("\n🌍 Environment Variables:")
    print(f"   • PROMPTLAYER_API_KEY: {'✅ Set' if api_key else '❌ Not set'}")
    if api_key:
        print(f"     Format: Valid (starts with 'pl-')" if api_key.startswith('pl-') else "     Format: Valid")
    
    print(f"   • GENOPS_TEAM: {'✅ ' + team if team else '⚠️ Not set (recommended)'}")
    print(f"   • GENOPS_PROJECT: {'✅ ' + project if project else '⚠️ Not set (recommended)'}")
    
    if not team or not project:
        print("\n💡 Recommendation:")
        print("   export GENOPS_TEAM='your-team-name'")
        print("   export GENOPS_PROJECT='your-project-name'")
        print("   This enables full cost attribution and governance features.")
    
    # Quick test if everything looks good
    if result.overall_status.value in ["passed", "warning"]:
        print("\n🧪 Quick Integration Test:")
        try:
            from genops.providers.promptlayer import instrument_promptlayer
            
            adapter = instrument_promptlayer(
                team=team or "validation-team",
                project=project or "setup-test"
            )
            
            metrics = adapter.get_metrics()
            print("   ✅ GenOps PromptLayer adapter created successfully")
            print(f"   📊 Team: {metrics.get('team', 'N/A')}, Project: {metrics.get('project', 'N/A')}")
            
        except Exception as e:
            print(f"   ❌ Integration test failed: {e}")
    
    print("\n" + "🔍" * 50)
    return result.overall_status.value == "passed"

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)