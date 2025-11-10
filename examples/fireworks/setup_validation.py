#!/usr/bin/env python3
"""
Fireworks AI + GenOps Setup Validation

Comprehensive validation script for Fireworks AI integration with GenOps governance.
Verifies API authentication, model access, performance, and provides diagnostics.

Usage:
    python setup_validation.py

Environment Variables:
    FIREWORKS_API_KEY: Your Fireworks AI API key
    GENOPS_TEAM: Team name for cost attribution
    GENOPS_PROJECT: Project name for tracking
    GENOPS_ENVIRONMENT: Environment (dev/staging/prod)
"""

import os
import sys

try:
    from genops.providers.fireworks_pricing import FireworksPricingCalculator
    from genops.providers.fireworks_validation import validate_fireworks_setup
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install GenOps: pip install genops-ai[fireworks]")
    sys.exit(1)


def main():
    """Run comprehensive Fireworks AI + GenOps validation."""
    print("🔧 Fireworks AI + GenOps Setup Validation")
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
    print("📋 Configuration:")
    print(f"   Team: {config['team']}")
    print(f"   Project: {config['project']}")
    print(f"   Environment: {config['environment']}")
    print(f"   Daily Budget: ${config['daily_budget_limit']}")
    print(f"   API Key: {'✅ Set' if os.getenv('FIREWORKS_API_KEY') else '❌ Not set'}")

    # Run validation
    try:
        result = validate_fireworks_setup(
            config=config,
            print_results=True
        )

        # Additional analysis if validation passes
        if result.is_valid and result.model_access:
            print("\n" + "=" * 60)
            print("🎯 Model Recommendations & Cost Analysis")
            print("=" * 60)

            pricing_calc = FireworksPricingCalculator()

            # Show cost comparison for accessible models
            accessible_models = result.model_access[:5]  # Top 5 accessible
            comparisons = pricing_calc.compare_models(accessible_models, estimated_tokens=1000)

            print("\n💰 Cost Comparison (1000 tokens):")
            for comp in comparisons:
                print(f"   {comp['model'].split('/')[-1]}")
                print(f"      Cost: ${comp['estimated_cost']:.4f} ({comp['tier']} tier)")
                print(f"      Context: {comp['context_length']:,} tokens")
                if comp.get('batch_cost'):
                    print(f"      Batch: ${comp['batch_cost']:.4f} (saves ${comp['batch_savings']:.4f})")
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

                if rec.recommended_model:
                    print(f"   {complexity.title()}: {description}")
                    print(f"      → {rec.recommended_model.split('/')[-1]}")
                    print(f"      → ${rec.estimated_cost:.4f} per operation")
                    print()

            # Show cost analysis for projected usage
            print("📊 Cost Analysis (1000 operations/day):")
            analysis = pricing_calc.analyze_costs(
                operations_per_day=1000,
                avg_tokens_per_operation=500,
                model=accessible_models[0],  # Use first accessible model
                days_to_analyze=30,
                batch_percentage=0.3  # 30% batch processing
            )

            print(f"   Model: {analysis['current_model'].split('/')[-1]}")
            print(f"   Daily cost: ${analysis['cost_analysis']['daily_cost']:.2f}")
            print(f"   Monthly cost: ${analysis['cost_analysis']['monthly_cost']:.2f}")
            print(f"   Cost per operation: ${analysis['cost_analysis']['cost_per_operation']:.4f}")
            print(f"   Batch savings: ${analysis['optimization']['batch_optimization_potential']:.2f}")

            if analysis['optimization']['best_alternative']:
                alt = analysis['optimization']['best_alternative']
                print(f"\n   💡 Alternative: {alt['model'].split('/')[-1]}")
                print(f"   Potential monthly savings: ${analysis['optimization']['potential_monthly_savings']:.2f}")

        # Final status
        print("\n" + "=" * 60)
        if result.is_valid:
            print("✅ VALIDATION COMPLETE - Ready for Fireworks AI operations!")
            print("\n🚀 Next Steps:")
            print("   1. Run: python basic_tracking.py")
            print("   2. Try: python cost_optimization.py")
            print("   3. Explore: python advanced_features.py")
            print("   4. Performance: Expect 4x faster inference with Fireattention!")
        else:
            print("❌ VALIDATION FAILED - Please resolve issues above")
            print("\n🔧 Common fixes:")
            print("   1. Set FIREWORKS_API_KEY environment variable")
            print("   2. Install: pip install fireworks-ai")
            print("   3. Verify API key in Fireworks AI dashboard")
            print("   4. Check network connectivity")

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
