#!/usr/bin/env python3
"""
AutoGen + GenOps Setup Validation

30-second validation to ensure your AutoGen + GenOps integration is ready.
This should be your first step before using any other AutoGen examples.

Features:
    - Complete environment validation in under 30 seconds
    - Checks AutoGen installation and version compatibility
    - Validates API keys and connectivity
    - Tests GenOps integration readiness
    - Provides actionable fix suggestions for any issues
    - CI/CD pipeline friendly with exit codes

Usage:
    python examples/autogen/setup_validation.py
    
    # For CI/CD (returns exit code 0 for success, 1 for failure)
    python examples/autogen/setup_validation.py --ci

Requirements:
    pip install genops[autogen]
"""

import sys
import os
import argparse
from typing import Dict, Any

def quick_validate_environment() -> Dict[str, Any]:
    """Ultra-fast environment validation for immediate feedback."""
    result = {
        "success": True,
        "issues": [],
        "fixes": [],
        "score": 100
    }
    
    print("🔍 AutoGen + GenOps Quick Validation")
    print("=" * 40)
    
    # Check 1: Python version (2 seconds max)
    print("📋 Checking Python version...", end=" ")
    if sys.version_info < (3, 8):
        result["success"] = False
        result["issues"].append("Python 3.8+ required")
        result["fixes"].append("Upgrade Python: https://python.org/downloads")
        result["score"] -= 30
        print("❌ FAIL")
    else:
        print("✅ PASS")
    
    # Check 2: AutoGen installation (5 seconds max)
    print("🤖 Checking AutoGen installation...", end=" ")
    try:
        import autogen
        version = getattr(autogen, '__version__', 'unknown')
        print(f"✅ PASS ({version})")
    except ImportError:
        result["success"] = False
        result["issues"].append("AutoGen not installed")
        result["fixes"].append("Install AutoGen: pip install pyautogen")
        result["score"] -= 25
        print("❌ FAIL")
    
    # Check 3: GenOps installation (5 seconds max)
    print("⚙️  Checking GenOps installation...", end=" ")
    try:
        from genops.providers.autogen import validate_autogen_setup
        print("✅ PASS")
    except ImportError:
        result["success"] = False
        result["issues"].append("GenOps not installed")
        result["fixes"].append("Install GenOps: pip install genops")
        result["score"] -= 25
        print("❌ FAIL")
        return result  # Can't continue without GenOps
    
    # Check 4: API Keys (3 seconds max)
    print("🔑 Checking API keys...", end=" ")
    api_keys = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'COHERE_API_KEY']
    found_keys = [key for key in api_keys if os.getenv(key)]
    
    if not found_keys:
        result["issues"].append("No API keys found")
        result["fixes"].append("Set at least one API key: export OPENAI_API_KEY=your_key")
        result["score"] -= 15
        print("⚠️  WARN")
    else:
        print(f"✅ PASS ({len(found_keys)} keys)")
    
    # Check 5: Basic adapter creation (10 seconds max)
    print("🔧 Testing GenOps integration...", end=" ")
    try:
        from genops.providers.autogen import GenOpsAutoGenAdapter
        adapter = GenOpsAutoGenAdapter(team="validation-test", project="quick-test")
        print("✅ PASS")
    except Exception as e:
        result["success"] = False
        result["issues"].append(f"GenOps integration error: {str(e)}")
        result["fixes"].append("Check GenOps installation: pip install --upgrade genops")
        result["score"] -= 20
        print("❌ FAIL")
    
    return result

def comprehensive_validate() -> Dict[str, Any]:
    """Comprehensive validation using GenOps built-in validation."""
    print("\n🔬 Running comprehensive validation...")
    
    try:
        from genops.providers.autogen import validate_autogen_setup, print_validation_result
        
        result = validate_autogen_setup(
            team="validation-test",
            project="comprehensive-test",
            verify_connectivity=True,
            run_performance_tests=False  # Keep it under 30 seconds total
        )
        
        # Convert to our format
        return {
            "success": result.success,
            "score": result.overall_score,
            "issues": [issue.title for issue in result.issues if issue.severity == "error"],
            "warnings": [issue.title for issue in result.issues if issue.severity == "warning"],
            "full_result": result
        }
        
    except Exception as e:
        return {
            "success": False,
            "score": 0,
            "issues": [f"Comprehensive validation failed: {str(e)}"],
            "fixes": ["Check GenOps installation"],
            "full_result": None
        }

def print_results(quick_result: Dict[str, Any], comprehensive_result: Dict[str, Any] = None):
    """Print validation results in a user-friendly format."""
    
    print("\n" + "=" * 40)
    print("📊 VALIDATION RESULTS")
    print("=" * 40)
    
    # Overall status
    if quick_result["success"] and (not comprehensive_result or comprehensive_result["success"]):
        print("🎉 STATUS: READY FOR AUTOGEN + GENOPS!")
        status_color = "🟢"
    elif quick_result["score"] > 70:
        print("⚠️  STATUS: MOSTLY READY (minor issues)")
        status_color = "🟡"
    else:
        print("❌ STATUS: NOT READY (critical issues)")
        status_color = "🔴"
    
    print(f"{status_color} SCORE: {quick_result['score']:.0f}/100")
    
    # Issues and fixes
    if quick_result["issues"]:
        print(f"\n❌ ISSUES FOUND ({len(quick_result['issues'])}):")
        for i, issue in enumerate(quick_result["issues"], 1):
            print(f"   {i}. {issue}")
    
    if quick_result.get("fixes"):
        print(f"\n💡 QUICK FIXES:")
        for i, fix in enumerate(quick_result["fixes"], 1):
            print(f"   {i}. {fix}")
    
    if comprehensive_result and comprehensive_result.get("warnings"):
        print(f"\n⚠️  WARNINGS ({len(comprehensive_result['warnings'])}):")
        for warning in comprehensive_result["warnings"][:3]:  # Show top 3
            print(f"   • {warning}")
    
    # Next steps
    print("\n🚀 NEXT STEPS:")
    if quick_result["success"]:
        print("   1. Try: python examples/autogen/basic_conversation_tracking.py")
        print("   2. Read: docs/quickstart/autogen-quickstart.md")
        print("   3. Explore: examples/autogen/ for more patterns")
    else:
        print("   1. Fix the issues listed above")
        print("   2. Run this validation again")
        print("   3. Get help: https://github.com/KoshiHQ/GenOps-AI/issues")
    
    print("=" * 40)

def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(description="Validate AutoGen + GenOps setup")
    parser.add_argument("--ci", action="store_true", help="CI/CD mode (exit codes only)")
    parser.add_argument("--quick", action="store_true", help="Quick validation only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    if args.ci:
        print("Running CI validation...", end="")
    
    # Always run quick validation (under 30 seconds)
    quick_result = quick_validate_environment()
    
    comprehensive_result = None
    if not args.quick and quick_result["success"]:
        comprehensive_result = comprehensive_validate()
    
    # CI mode: just return exit code
    if args.ci:
        success = quick_result["success"] and (not comprehensive_result or comprehensive_result["success"])
        print(" PASS" if success else " FAIL")
        sys.exit(0 if success else 1)
    
    # Interactive mode: show detailed results
    print_results(quick_result, comprehensive_result)
    
    # Verbose mode: show full comprehensive results
    if args.verbose and comprehensive_result and comprehensive_result["full_result"]:
        print("\n" + "=" * 40)
        print("🔬 DETAILED VALIDATION RESULTS")
        print("=" * 40)
        from genops.providers.autogen import print_validation_result
        print_validation_result(comprehensive_result["full_result"], verbose=True)
    
    # Exit with appropriate code
    success = quick_result["success"] and (not comprehensive_result or comprehensive_result["success"])
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()