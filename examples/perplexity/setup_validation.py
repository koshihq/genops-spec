#!/usr/bin/env python3
"""
Perplexity AI Setup Validation Example

This example demonstrates comprehensive setup validation for Perplexity AI integration
with GenOps governance, including API connectivity, model access, and governance
configuration verification.

Usage:
    python setup_validation.py

Prerequisites:
    pip install genops[perplexity]
    export PERPLEXITY_API_KEY="pplx-your-api-key"
    export GENOPS_TEAM="your-team-name" (optional)
    export GENOPS_PROJECT="your-project-name" (optional)

Expected Output:
    Complete validation report with:
    - ✅ Dependencies and API connectivity confirmed
    - ⚠️ Governance configuration recommendations
    - 🔍 Search feature capabilities validation
    - 📋 Actionable fix suggestions for any issues

Learning Objectives:
    - Understand Perplexity API requirements and setup
    - Learn GenOps governance configuration options
    - Practice troubleshooting common setup issues
    - Validate search-specific features and capabilities

Time Required: ~2 minutes
"""

import os
import sys
from typing import Dict, Any


def validate_prerequisites() -> bool:
    """Check if basic prerequisites are met."""
    print("🔍 Checking Prerequisites...")
    
    prerequisites_met = True
    
    # Check GenOps installation
    try:
        import genops
        print("  ✅ GenOps package installed")
    except ImportError:
        print("  ❌ GenOps package not found")
        print("     Fix: pip install genops[perplexity]")
        prerequisites_met = False
    
    # Check OpenAI client (required for Perplexity)
    try:
        import openai
        print("  ✅ OpenAI client available")
    except ImportError:
        print("  ❌ OpenAI client not found")
        print("     Fix: pip install openai")
        prerequisites_met = False
    
    # Check API key
    if os.getenv('PERPLEXITY_API_KEY'):
        print("  ✅ PERPLEXITY_API_KEY configured")
    else:
        print("  ⚠️ PERPLEXITY_API_KEY not set")
        print("     Fix: export PERPLEXITY_API_KEY='pplx-your-api-key'")
        print("     Note: Get your key from https://www.perplexity.ai/settings/api")
    
    return prerequisites_met


def run_validation_example() -> Dict[str, Any]:
    """Run comprehensive Perplexity setup validation."""
    print("\n🔬 Perplexity AI Setup Validation Example")
    print("=" * 55)
    
    if not validate_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above and try again.")
        return {'success': False, 'error': 'prerequisites_not_met'}
    
    try:
        from genops.providers.perplexity_validation import validate_setup, print_validation_result
        
        print("\n🧪 Running comprehensive validation...")
        
        # Run complete validation
        result = validate_setup()
        
        # Print detailed results
        print_validation_result(result)
        
        # Return summary for further use
        return {
            'success': True,
            'validation_result': result,
            'is_valid': result.is_valid,
            'error_count': result.error_count,
            'warning_count': result.warning_count,
            'recommendations': _extract_recommendations(result)
        }
        
    except ImportError as e:
        print(f"❌ GenOps Perplexity provider not available: {e}")
        print("   Fix: pip install genops[perplexity]")
        return {'success': False, 'error': 'import_error', 'details': str(e)}
    
    except Exception as e:
        print(f"❌ Validation failed with unexpected error: {e}")
        return {'success': False, 'error': 'validation_error', 'details': str(e)}


def _extract_recommendations(validation_result) -> Dict[str, Any]:
    """Extract key recommendations from validation result."""
    recommendations = {
        'immediate_actions': [],
        'optional_improvements': [],
        'next_steps': []
    }
    
    # Extract immediate actions (errors)
    error_issues = [issue for issue in validation_result.issues if issue.level.value == 'error']
    for issue in error_issues:
        if issue.fix_suggestions:
            recommendations['immediate_actions'].extend(issue.fix_suggestions[:2])
    
    # Extract optional improvements (warnings)
    warning_issues = [issue for issue in validation_result.issues if issue.level.value == 'warning']
    for issue in warning_issues:
        if issue.fix_suggestions:
            recommendations['optional_improvements'].extend(issue.fix_suggestions[:1])
    
    # Determine next steps
    if validation_result.error_count > 0:
        recommendations['next_steps'] = [
            "Fix critical errors before proceeding",
            "Re-run validation to confirm fixes",
            "Try basic_search.py example once setup is complete"
        ]
    elif validation_result.warning_count > 0:
        recommendations['next_steps'] = [
            "Basic functionality available - try examples",
            "Address warnings for optimal performance", 
            "Configure governance settings for production use"
        ]
    else:
        recommendations['next_steps'] = [
            "✅ Setup is complete and ready for use!",
            "Try basic_search.py for your first search",
            "Explore advanced examples for production patterns"
        ]
    
    return recommendations


def demonstrate_interactive_wizard():
    """Demonstrate the interactive setup wizard."""
    print("\n🧙‍♂️ Interactive Setup Wizard Demo")
    print("-" * 40)
    print("The interactive wizard helps configure Perplexity + GenOps step by step.")
    print("This is especially useful for first-time setup or complex configurations.")
    print()
    
    user_input = input("Would you like to run the interactive setup wizard? [y/N]: ").strip().lower()
    
    if user_input in ['y', 'yes']:
        try:
            from genops.providers.perplexity_validation import interactive_setup_wizard
            config = interactive_setup_wizard()
            
            print(f"\n📋 Wizard completed! Generated configuration with {len(config)} settings.")
            return config
            
        except ImportError:
            print("❌ Interactive wizard not available. Ensure GenOps is properly installed.")
            return None
        except KeyboardInterrupt:
            print("\n⏹️ Wizard cancelled by user.")
            return None
        except Exception as e:
            print(f"❌ Wizard error: {e}")
            return None
    else:
        print("⏩ Skipping interactive wizard.")
        return None


def main():
    """Main example execution."""
    print("🚀 Perplexity AI + GenOps Setup Validation")
    print("=" * 50)
    print()
    print("This example validates your Perplexity AI integration setup,")
    print("checks API connectivity, and provides actionable recommendations.")
    print()
    
    # Run validation
    result = run_validation_example()
    
    if result['success']:
        print(f"\n📊 Validation Summary:")
        print(f"   Setup Valid: {'✅ Yes' if result['is_valid'] else '❌ No'}")
        print(f"   Errors: {result['error_count']}")
        print(f"   Warnings: {result['warning_count']}")
        
        if result['recommendations']['next_steps']:
            print(f"\n🎯 Recommended Next Steps:")
            for i, step in enumerate(result['recommendations']['next_steps'], 1):
                print(f"   {i}. {step}")
    
    # Interactive wizard demo
    wizard_config = demonstrate_interactive_wizard()
    
    # Provide helpful next steps
    print(f"\n📚 What's Next?")
    print("   • Try basic_search.py for your first search")
    print("   • Explore cost_optimization.py for cost management")
    print("   • Read docs/perplexity-quickstart.md for complete guide")
    print("   • Check examples/perplexity/ for more advanced patterns")
    
    return result


if __name__ == "__main__":
    try:
        result = main()
        exit_code = 0 if result.get('success', False) else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️ Example cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        sys.exit(1)