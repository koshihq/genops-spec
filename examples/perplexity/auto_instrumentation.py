#!/usr/bin/env python3
"""
Perplexity AI Auto-Instrumentation Example

This example demonstrates zero-code auto-instrumentation for Perplexity AI,
allowing existing code to work unchanged while adding GenOps governance,
cost tracking, and observability.

Usage:
    python auto_instrumentation.py

Prerequisites:
    pip install genops[perplexity]
    export PERPLEXITY_API_KEY="pplx-your-api-key"
    export GENOPS_TEAM="your-team-name"
    export GENOPS_PROJECT="your-project-name"

Expected Output:
    - ✅ Existing Perplexity code works unchanged
    - 📊 Automatic governance and cost tracking
    - 🔍 Zero-code search session management
    - 💰 Transparent cost attribution and reporting

Learning Objectives:
    - Enable governance without changing existing code
    - Understand auto-instrumentation capabilities
    - Learn transparent cost tracking mechanisms
    - Practice zero-configuration governance setup

Time Required: ~3 minutes
"""

import os
import time


def demonstrate_traditional_usage():
    """Show how traditional Perplexity code works unchanged."""
    print("📱 Traditional Perplexity Usage (Before GenOps)")
    print("-" * 50)
    print("This is how you normally use Perplexity with the OpenAI client:")
    print()

    code_example = '''
import openai

client = openai.OpenAI(
    api_key="pplx-your-api-key",
    base_url="https://api.perplexity.ai"
)

response = client.chat.completions.create(
    model="sonar-pro",
    messages=[{"role": "user", "content": "AI trends 2024"}]
)

print(response.choices[0].message.content)
'''

    print(code_example)
    print("❌ Problems with traditional approach:")
    print("   • No cost tracking or attribution")
    print("   • No governance or budget controls")
    print("   • No team/project visibility")
    print("   • No observability or monitoring")


def demonstrate_auto_instrumentation():
    """Demonstrate zero-code auto-instrumentation."""
    print("\n🚀 Zero-Code Auto-Instrumentation with GenOps")
    print("=" * 55)
    print("Add ONE line to enable governance for all Perplexity operations!")
    print()

    try:
        # Step 1: Enable auto-instrumentation (THE ONLY CHANGE NEEDED)
        print("🔧 Step 1: Enable auto-instrumentation...")

        from genops.providers.perplexity import auto_instrument

        # This ONE line adds governance to all Perplexity operations
        adapter = auto_instrument(
            team=os.getenv('GENOPS_TEAM', 'auto-instrumented-team'),
            project=os.getenv('GENOPS_PROJECT', 'zero-code-example'),
            environment='development',
            daily_budget_limit=25.0,
            governance_policy='advisory'
        )

        print("✅ Auto-instrumentation enabled!")
        print(f"   Team: {adapter.team}")
        print(f"   Project: {adapter.project}")
        print(f"   Budget: ${adapter.daily_budget_limit}/day")
        print(f"   Policy: {adapter.governance_policy}")

        # Step 2: Use existing code patterns - NO CHANGES NEEDED
        print("\n🔧 Step 2: Use existing Perplexity code (unchanged)...")

        # This is exactly how you'd normally use Perplexity
        # But now it has governance and cost tracking!
        traditional_perplexity_code(adapter)

        # Step 3: Show governance benefits
        print("\n📊 Step 3: Automatic governance benefits...")
        show_governance_benefits(adapter)

        return True

    except ImportError as e:
        print(f"❌ GenOps not available: {e}")
        print("   Fix: pip install genops[perplexity]")
        return False

    except Exception as e:
        print(f"❌ Auto-instrumentation failed: {e}")
        return False


def traditional_perplexity_code(adapter):
    """Traditional Perplexity code that now has governance."""
    print("   Running traditional Perplexity patterns...")

    # Traditional usage pattern - works exactly the same!
    try:
        import openai

        # Your existing Perplexity client setup (unchanged)
        client = openai.OpenAI(
            api_key=os.getenv('PERPLEXITY_API_KEY'),
            base_url="https://api.perplexity.ai"
        )

        # Your existing search requests (unchanged)
        queries = [
            "What are the latest AI breakthroughs in 2024?",
            "Best practices for cloud security",
            "Future of renewable energy technology"
        ]

        for i, query in enumerate(queries, 1):
            print(f"   🔍 Query {i}: {query[:40]}...")

            # Existing code pattern - NO CHANGES NEEDED
            response = client.chat.completions.create(
                model="sonar",
                messages=[{"role": "user", "content": query}],
                max_tokens=150
            )

            result = response.choices[0].message.content
            tokens = response.usage.total_tokens if hasattr(response, 'usage') else 150

            print(f"       Response: {result[:80]}...")
            print(f"       Tokens: {tokens}")

            # Small delay between requests
            time.sleep(0.5)

        print("   ✅ All traditional code executed successfully")
        print("   🎯 But now with automatic governance and cost tracking!")

    except Exception as e:
        print(f"   ❌ Traditional code execution failed: {e}")
        # Fallback to direct adapter usage
        fallback_demonstration(adapter)


def fallback_demonstration(adapter):
    """Fallback demonstration using adapter directly."""
    print("   📱 Fallback: Using GenOps adapter directly...")

    try:
        queries = [
            "What is artificial intelligence?",
            "How does machine learning work?"
        ]

        with adapter.track_search_session("fallback_demo") as session:
            for query in queries:
                result = adapter.search_with_governance(
                    query=query,
                    model="sonar",
                    max_tokens=100,
                    session_id=session.session_id
                )

                print(f"   ✅ Search completed: {result.tokens_used} tokens, ${result.cost:.6f}")

    except Exception as e:
        print(f"   ❌ Fallback demonstration failed: {e}")


def show_governance_benefits(adapter):
    """Show the governance benefits added by auto-instrumentation."""

    # Cost tracking
    cost_summary = adapter.get_cost_summary()

    print("💰 Automatic Cost Intelligence:")
    print(f"   Daily Spend: ${cost_summary['daily_costs']:.6f}")
    print(f"   Budget Used: {cost_summary['daily_budget_utilization']:.1f}%")
    print(f"   Team: {cost_summary['team']}")
    print(f"   Project: {cost_summary['project']}")

    # Show active sessions
    print("\n📊 Session Management:")
    print(f"   Active Sessions: {cost_summary['active_sessions']}")
    print(f"   Environment: {cost_summary['environment']}")
    print(f"   Governance: {'✅ Enabled' if cost_summary['governance_enabled'] else '❌ Disabled'}")

    # Cost optimization insights
    try:
        analysis = adapter.get_search_cost_analysis(projected_queries=50)

        print("\n🎯 Cost Optimization Insights:")
        print(f"   Cost per search: ${analysis['current_cost_structure']['cost_per_query']:.6f}")

        if analysis['optimization_opportunities']:
            top_opt = analysis['optimization_opportunities'][0]
            print(f"   💡 Top optimization: {top_opt['optimization_type']}")
            print(f"   💰 Potential savings: ${top_opt['potential_savings_total']:.4f}")

    except Exception as e:
        print(f"   Note: Detailed analysis unavailable: {str(e)[:50]}")


def demonstrate_configuration_options():
    """Show different auto-instrumentation configuration options."""
    print("\n⚙️ Auto-Instrumentation Configuration Options")
    print("=" * 50)

    configurations = [
        {
            'name': 'Development Mode',
            'config': {
                'team': 'dev-team',
                'environment': 'development',
                'governance_policy': 'advisory',
                'daily_budget_limit': 10.0
            },
            'description': 'Minimal governance for development'
        },
        {
            'name': 'Production Mode',
            'config': {
                'team': 'prod-team',
                'environment': 'production',
                'governance_policy': 'enforced',
                'daily_budget_limit': 100.0,
                'enable_cost_alerts': True
            },
            'description': 'Strict governance for production'
        },
        {
            'name': 'Enterprise Mode',
            'config': {
                'team': 'enterprise-team',
                'project': 'mission-critical',
                'environment': 'production',
                'governance_policy': 'strict',
                'daily_budget_limit': 500.0,
                'monthly_budget_limit': 10000.0,
                'customer_id': 'enterprise-123',
                'cost_center': 'ai-research'
            },
            'description': 'Maximum governance and attribution'
        }
    ]

    for config in configurations:
        print(f"\n🔧 {config['name']}:")
        print(f"   Description: {config['description']}")
        print("   Configuration:")

        for key, value in config['config'].items():
            print(f"     {key}: {value}")

        print("   Code example:")
        print("     auto_instrument(")
        for key, value in list(config['config'].items())[:3]:  # Show first 3
            if isinstance(value, str):
                print(f"         {key}='{value}',")
            else:
                print(f"         {key}={value},")
        if len(config['config']) > 3:
            print(f"         # ... and {len(config['config']) - 3} more options")
        print("     )")


def show_migration_guide():
    """Show how to migrate existing code to use auto-instrumentation."""
    print("\n🔄 Migration Guide: Adding GenOps to Existing Code")
    print("=" * 55)

    print("Step 1: Install GenOps")
    print("   pip install genops[perplexity]")
    print()

    print("Step 2: Add auto-instrumentation (at the top of your file)")
    print("   from genops.providers.perplexity import auto_instrument")
    print("   auto_instrument()  # Just add this line!")
    print()

    print("Step 3: Your existing code works unchanged!")
    print("   # No changes needed to your existing Perplexity code")
    print("   # Everything now has governance and cost tracking")
    print()

    print("📊 Benefits you get automatically:")
    print("   ✅ Cost tracking and attribution")
    print("   ✅ Team and project visibility")
    print("   ✅ Budget controls and alerts")
    print("   ✅ Performance monitoring")
    print("   ✅ Governance policy enforcement")
    print("   ✅ Session management")
    print()

    print("🎯 Best Practices:")
    print("   • Set GENOPS_TEAM and GENOPS_PROJECT environment variables")
    print("   • Configure appropriate budget limits for your use case")
    print("   • Use 'advisory' policy for development, 'enforced' for production")
    print("   • Monitor cost summaries regularly")


def main():
    """Main example execution."""
    print("🚀 Perplexity AI Auto-Instrumentation Example")
    print("=" * 50)
    print()
    print("This example shows how to add GenOps governance to existing")
    print("Perplexity AI code with zero changes to your existing patterns.")
    print()

    # Show traditional approach
    demonstrate_traditional_usage()

    # Show auto-instrumentation
    success = demonstrate_auto_instrumentation()

    if success:
        # Show configuration options
        demonstrate_configuration_options()

        # Show migration guide
        show_migration_guide()

        print("\n🎉 Auto-instrumentation example completed!")
        print("\n📚 Next Steps:")
        print("   • Apply auto_instrument() to your existing code")
        print("   • Try advanced_search.py for more complex patterns")
        print("   • Explore cost_optimization.py for budget management")

        return True
    else:
        print("\n❌ Auto-instrumentation example failed")
        print("   • Check prerequisites and try setup_validation.py")
        return False


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Example cancelled by user.")
        exit(1)
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        exit(1)
