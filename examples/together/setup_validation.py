#!/usr/bin/env python3
"""
Together AI + GenOps Setup Validation

Comprehensive validation script for Together AI integration with GenOps governance.
Verifies API authentication, model access, configuration, and provides diagnostics.

Usage:
    python setup_validation.py

Environment Variables:
    TOGETHER_API_KEY: Your Together AI API key
    GENOPS_TEAM: Team name for cost attribution
    GENOPS_PROJECT: Project name for tracking
    GENOPS_ENVIRONMENT: Environment (dev/staging/prod)
"""

import os
import sys
from typing import Dict, Any

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.genops.providers.together_validation import validate_together_setup
    from src.genops.providers.together_pricing import TogetherPricingCalculator
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)


def main():
    """Run comprehensive Together AI + GenOps validation."""
    print("🔧 Together AI + GenOps Setup Validation")
    print("=" * 50)
    
    # Gather configuration from environment
    config = {
        'team': os.getenv('GENOPS_TEAM', 'validation-team'),
        'project': os.getenv('GENOPS_PROJECT', 'setup-validation'),
        'environment': os.getenv('GENOPS_ENVIRONMENT', 'development'),
        'daily_budget_limit': 100.0,
        'monthly_budget_limit': 2000.0,
        'enable_governance': True,
        'enable_cost_alerts': True,
        'governance_policy': 'advisory'
    }
    
    # Show current configuration (safely)
    print(f"📋 Configuration:")
    print(f"   Team: {config['team']}")
    print(f"   Project: {config['project']}")
    print(f"   Environment: {config['environment']}")
    print(f"   Daily Budget: ${config['daily_budget_limit']}")
    print(f"   API Key: {'✅ Set' if os.getenv('TOGETHER_API_KEY') else '❌ Not set'}")
    
    # Run validation
    try:
        result = validate_together_setup(
            config=config,
            print_results=True
        )
        
        # Additional analysis if validation passes
        if result.is_valid and result.model_access:
            print("\n" + "=" * 60)
            print("🎯 Model Recommendations & Cost Analysis")
            print("=" * 60)
            
            pricing_calc = TogetherPricingCalculator()
            
            # Show cost comparison for accessible models
            accessible_models = result.model_access[:5]  # Top 5 accessible
            comparisons = pricing_calc.compare_models(accessible_models, estimated_tokens=1000)
            
            print("\n💰 Cost Comparison (1000 tokens):")
            for comp in comparisons:
                print(f"   {comp['model']}")
                print(f"      Cost: ${comp['estimated_cost']:.4f} ({comp['tier']} tier)")
                print(f"      Context: {comp['context_length']:,} tokens")
                print()
            
            # Show task-specific recommendations
            print("🧠 Model Recommendations by Task:")
            
            tasks = [
                ("simple", "Simple Q&A, basic chat"),
                ("moderate", "Analysis, code review, research"),
                ("complex", "Advanced reasoning, complex coding")
            ]
            
            for complexity, description in tasks:
                rec = pricing_calc.recommend_model(
                    task_complexity=complexity,
                    budget_per_operation=0.01,  # $0.01 budget
                    min_context_length=8192
                )
                
                if rec['recommended_model']:
                    print(f"   {complexity.title()}: {description}")
                    print(f"      → {rec['recommended_model']}")
                    print(f"      → ${rec['estimated_cost']:.4f} per operation")
                    print()
            
            # Show cost analysis for projected usage
            print("📊 Cost Analysis (1000 operations/day):")
            analysis = pricing_calc.analyze_costs(
                operations_per_day=1000,
                avg_tokens_per_operation=500,
                model=accessible_models[0],  # Use cheapest accessible model
                days_to_analyze=30
            )
            
            print(f"   Model: {analysis['current_model']}")
            print(f"   Daily cost: ${analysis['daily_cost']:.2f}")
            print(f"   Monthly cost: ${analysis['monthly_cost']:.2f}")
            print(f"   Cost per operation: ${analysis['cost_per_operation']:.4f}")
            
            if analysis['potential_savings']['best_alternative']:
                alt = analysis['potential_savings']['best_alternative']
                print(f"\n   💡 Alternative: {alt['model']}")
                print(f"   Potential monthly savings: ${analysis['potential_savings']['potential_monthly_savings']:.2f}")
        
        # Final status
        print("\n" + "=" * 60)
        if result.is_valid:
            print("✅ VALIDATION COMPLETE - Ready for Together AI operations!")
            print("\n🚀 Next Steps:")
            print("   1. Run: python basic_tracking.py")
            print("   2. Try: python cost_optimization.py")
            print("   3. Explore: python advanced_features.py")
        else:
            print("❌ VALIDATION FAILED - Please resolve issues above")
            print("\n🔧 Common fixes:")
            print("   1. Set TOGETHER_API_KEY environment variable")
            print("   2. Install: pip install together")
            print("   3. Verify API key in Together AI dashboard")
        
        return 0 if result.is_valid else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Validation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        print("Please check your configuration and try again")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)