#!/usr/bin/env python3
"""
Perplexity AI Cost Optimization Example

This example demonstrates cost optimization strategies for Perplexity AI including
intelligent model selection, search context optimization, budget management,
and comprehensive cost analysis with recommendations.

Usage:
    python cost_optimization.py

Prerequisites:
    pip install genops[perplexity]
    export PERPLEXITY_API_KEY="pplx-your-api-key"
    export GENOPS_TEAM="your-team-name"
    export GENOPS_PROJECT="your-project-name"

Expected Output:
    - 💰 Comprehensive cost analysis and optimization strategies
    - 📊 Model and context cost comparisons
    - 🎯 Budget management and enforcement demonstrations
    - 📈 Volume discount analysis and recommendations

Learning Objectives:
    - Master Perplexity's dual pricing model (tokens + requests)
    - Implement cost-aware model and context selection
    - Configure budget controls and cost optimization
    - Analyze volume pricing and optimization opportunities

Time Required: ~10 minutes
"""

import os
import time
from decimal import Decimal


def main():
    """Run comprehensive cost optimization example."""
    print("💰 Perplexity AI + GenOps Cost Optimization Example")
    print("=" * 55)
    print()
    print("This example demonstrates cost optimization strategies including")
    print("intelligent model selection, budget management, and volume analysis.")
    print()

    try:
        from genops.providers.perplexity import (
            GenOpsPerplexityAdapter,
            PerplexityModel,
            SearchContext,
        )
        from genops.providers.perplexity_pricing import PerplexityPricingCalculator

        print("🔧 Initializing cost-optimized Perplexity adapter...")

        # Cost-optimized adapter configuration
        adapter = GenOpsPerplexityAdapter(
            team=os.getenv('GENOPS_TEAM', 'cost-optimization-team'),
            project=os.getenv('GENOPS_PROJECT', 'cost-intelligence-demo'),
            environment='development',
            daily_budget_limit=100.0,  # Set budget for demonstrations
            monthly_budget_limit=2500.0,
            enable_governance=True,
            enable_cost_alerts=True,
            governance_policy='enforced',  # Enforce budget limits
            tags={
                'example': 'cost_optimization',
                'focus': 'cost_intelligence',
                'optimization_enabled': 'true'
            }
        )

        print("✅ Cost-optimized adapter configured")
        print(f"   Daily Budget: ${adapter.daily_budget_limit}")
        print(f"   Monthly Budget: ${adapter.monthly_budget_limit}")
        print(f"   Governance: {adapter.governance_policy}")
        print(f"   Cost Alerts: {'✅ Enabled' if adapter.enable_cost_alerts else '❌ Disabled'}")

        # Initialize pricing calculator for detailed analysis
        calculator = PerplexityPricingCalculator()

        # Run cost optimization demonstrations
        demonstrate_pricing_model(calculator)
        demonstrate_model_cost_comparison(adapter, calculator)
        demonstrate_context_optimization(adapter, calculator)
        demonstrate_budget_management(adapter)
        demonstrate_volume_analysis(adapter, calculator)
        demonstrate_cost_forecasting(calculator)

        # Show final optimization summary
        show_optimization_summary(adapter)

        print("\n🎉 Cost optimization example completed!")
        return True

    except ImportError as e:
        print(f"❌ GenOps Perplexity provider not available: {e}")
        print("   Fix: pip install genops[perplexity]")
        return False

    except Exception as e:
        print(f"❌ Cost optimization example failed: {e}")
        return False


def demonstrate_pricing_model(calculator):
    """Demonstrate Perplexity's dual pricing model."""
    print("\n💡 Understanding Perplexity's Dual Pricing Model")
    print("=" * 50)
    print("Perplexity charges both token costs AND request fees:")
    print("• Token costs: Based on model and token type (input/output/citations)")
    print("• Request fees: Based on search context depth (low/medium/high)")
    print()

    # Example pricing breakdown
    example_scenarios = [
        {
            'name': 'Simple Query',
            'model': 'sonar',
            'tokens': 500,
            'context': SearchContext.LOW,
            'description': 'Basic search with minimal context'
        },
        {
            'name': 'Research Query',
            'model': 'sonar-pro',
            'tokens': 1000,
            'context': SearchContext.HIGH,
            'description': 'Comprehensive research with citations'
        },
        {
            'name': 'Reasoning Query',
            'model': 'sonar-reasoning-pro',
            'tokens': 1500,
            'context': SearchContext.MEDIUM,
            'description': 'Complex reasoning with search'
        }
    ]

    print("📊 Pricing Examples:")
    for scenario in example_scenarios:
        # Calculate detailed cost breakdown
        breakdown = calculator.get_detailed_cost_breakdown(
            model=scenario['model'],
            tokens_used=scenario['tokens'],
            search_context=scenario['context']
        )

        print(f"\n   💰 {scenario['name']} ({scenario['description']}):")
        print(f"      Model: {scenario['model']} | Tokens: {scenario['tokens']} | Context: {scenario['context'].value}")
        print(f"      Token Cost: ${breakdown.token_cost:.6f}")
        print(f"      Request Cost: ${breakdown.request_cost:.6f}")
        print(f"      Total Cost: ${breakdown.total_cost:.6f}")
        print(f"      Cost per Token: ${breakdown.cost_per_token:.8f}")

    print("\n🎯 Key Insights:")
    print("   • Token costs vary significantly by model (1-15x difference)")
    print("   • Request fees depend on search context depth")
    print("   • Total cost = Token cost + Request fee")
    print("   • Optimization requires balancing both components")


def demonstrate_model_cost_comparison(adapter, calculator):
    """Compare costs across different Perplexity models."""
    print("\n🤖 Model Cost Comparison")
    print("=" * 30)

    models_to_compare = [
        (PerplexityModel.SONAR, "Cost-effective general search"),
        (PerplexityModel.SONAR_PRO, "Enhanced accuracy and citations"),
        (PerplexityModel.SONAR_REASONING, "Basic reasoning capabilities")
    ]

    test_query = "Explain the benefits of renewable energy"
    comparison_results = []

    print(f"🔍 Testing query: \"{test_query}\"")
    print("📊 Model comparison results:")

    with adapter.track_search_session("model_cost_comparison") as session:
        for model, description in models_to_compare:
            print(f"\n   🧠 Testing {model.value.upper()}:")
            print(f"      Description: {description}")

            try:
                start_time = time.time()

                result = adapter.search_with_governance(
                    query=test_query,
                    model=model,
                    search_context=SearchContext.MEDIUM,
                    session_id=session.session_id,
                    max_tokens=200,  # Consistent for comparison
                    comparison_test=True
                )

                execution_time = time.time() - start_time

                comparison_results.append({
                    'model': model.value,
                    'cost': result.cost,
                    'tokens': result.tokens_used,
                    'citations': len(result.citations),
                    'time': execution_time,
                    'cost_per_token': result.cost / result.tokens_used if result.tokens_used > 0 else 0,
                    'response_quality': len(result.response)
                })

                print(f"      Cost: ${result.cost:.6f}")
                print(f"      Tokens: {result.tokens_used}")
                print(f"      Citations: {len(result.citations)}")
                print(f"      Time: {execution_time:.2f}s")
                print(f"      Cost/Token: ${result.cost / result.tokens_used:.8f}")

            except Exception as e:
                print(f"      ❌ Test failed: {str(e)[:50]}")

        # Analysis and recommendations
        if len(comparison_results) > 1:
            print("\n📈 Cost Comparison Analysis:")

            # Find cheapest and most expensive
            cheapest = min(comparison_results, key=lambda x: x['cost'])
            most_expensive = max(comparison_results, key=lambda x: x['cost'])

            cost_difference = most_expensive['cost'] - cheapest['cost']
            cost_ratio = most_expensive['cost'] / cheapest['cost'] if cheapest['cost'] > 0 else 0

            print(f"   💸 Cheapest: {cheapest['model']} (${cheapest['cost']:.6f})")
            print(f"   💰 Most Expensive: {most_expensive['model']} (${most_expensive['cost']:.6f})")
            print(f"   📊 Cost Difference: ${cost_difference:.6f} ({cost_ratio:.1f}x)")

            # Best value analysis
            best_value = max(comparison_results, key=lambda x: x['citations'] / x['cost'] if x['cost'] > 0 else 0)
            print(f"   🏆 Best Value: {best_value['model']} ({best_value['citations']} citations per ${best_value['cost']:.6f})")


def demonstrate_context_optimization(adapter, calculator):
    """Demonstrate search context optimization for cost savings."""
    print("\n🎯 Search Context Optimization")
    print("=" * 35)
    print("Search context affects request fees and result quality:")

    contexts = [SearchContext.LOW, SearchContext.MEDIUM, SearchContext.HIGH]
    query = "Best practices for database optimization"

    context_analysis = []

    print(f"\n🔍 Testing contexts with query: \"{query[:40]}...\"")

    with adapter.track_search_session("context_optimization") as session:
        for context in contexts:
            print(f"\n   📊 {context.value.upper()} Context:")

            try:
                # Calculate cost beforehand for comparison
                estimated_cost = calculator.estimate_search_cost(
                    model="sonar",
                    estimated_tokens=300,
                    search_context=context
                )

                result = adapter.search_with_governance(
                    query=query,
                    model=PerplexityModel.SONAR,
                    search_context=context,
                    session_id=session.session_id,
                    max_tokens=300
                )

                # Get detailed breakdown
                breakdown = calculator.get_detailed_cost_breakdown(
                    model="sonar",
                    tokens_used=result.tokens_used,
                    search_context=context
                )

                context_analysis.append({
                    'context': context.value,
                    'token_cost': breakdown.token_cost,
                    'request_cost': breakdown.request_cost,
                    'total_cost': breakdown.total_cost,
                    'citations': len(result.citations),
                    'response_length': len(result.response)
                })

                print(f"      Token Cost: ${breakdown.token_cost:.6f}")
                print(f"      Request Cost: ${breakdown.request_cost:.6f}")
                print(f"      Total Cost: ${breakdown.total_cost:.6f}")
                print(f"      Citations: {len(result.citations)}")
                print(f"      Response Length: {len(result.response)} chars")

            except Exception as e:
                print(f"      ❌ Context test failed: {str(e)[:50]}")

        # Context optimization recommendations
        if len(context_analysis) >= 2:
            print("\n🎯 Context Optimization Insights:")

            low_context = next((c for c in context_analysis if c['context'] == 'low'), None)
            high_context = next((c for c in context_analysis if c['context'] == 'high'), None)

            if low_context and high_context:
                request_cost_increase = high_context['request_cost'] - low_context['request_cost']
                citation_increase = high_context['citations'] - low_context['citations']

                print("   📈 HIGH vs LOW context:")
                print(f"      Request cost increase: ${request_cost_increase:.6f}")
                print(f"      Additional citations: {citation_increase}")

                if citation_increase > 0:
                    cost_per_additional_citation = request_cost_increase / citation_increase
                    print(f"      Cost per additional citation: ${cost_per_additional_citation:.6f}")

                print("\n   💡 Recommendations:")
                if request_cost_increase < Decimal('0.001'):
                    print("      • Context cost difference is minimal - use HIGH for better results")
                elif citation_increase > 3:
                    print(f"      • HIGH context provides good value with {citation_increase} more citations")
                else:
                    print("      • Consider MEDIUM context for balanced cost/quality")


def demonstrate_budget_management(adapter):
    """Demonstrate budget management and enforcement."""
    print("\n🏦 Budget Management and Enforcement")
    print("=" * 40)

    # Show current budget status
    cost_summary = adapter.get_cost_summary()

    print("💰 Current Budget Status:")
    print(f"   Daily Spend: ${cost_summary['daily_costs']:.6f}")
    print(f"   Daily Limit: ${cost_summary['daily_budget_limit']}")
    print(f"   Utilization: {cost_summary['daily_budget_utilization']:.1f}%")
    print(f"   Remaining: ${cost_summary['daily_budget_limit'] - cost_summary['daily_costs']:.4f}")

    # Demonstrate budget-aware operations
    print("\n🎯 Budget-Aware Search Demonstration:")

    with adapter.track_search_session("budget_management") as session:
        # Perform searches while monitoring budget
        test_queries = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "Explain neural networks",
            "What are the applications of AI?"
        ]

        successful_searches = 0
        budget_blocked_searches = 0

        for i, query in enumerate(test_queries, 1):
            print(f"\n   🔍 Search {i}: {query[:30]}...")

            try:
                # Check budget before search
                pre_search_summary = adapter.get_cost_summary()
                print(f"      Pre-search budget: {pre_search_summary['daily_budget_utilization']:.1f}% used")

                result = adapter.search_with_governance(
                    query=query,
                    model=PerplexityModel.SONAR,
                    search_context=SearchContext.LOW,  # Use low cost for demo
                    session_id=session.session_id,
                    max_tokens=100  # Limit tokens to control cost
                )

                successful_searches += 1
                print(f"      ✅ Completed: ${result.cost:.6f} cost")

                # Show updated budget
                post_search_summary = adapter.get_cost_summary()
                print(f"      Post-search budget: {post_search_summary['daily_budget_utilization']:.1f}% used")

            except Exception as e:
                if "budget" in str(e).lower():
                    budget_blocked_searches += 1
                    print(f"      🚫 Blocked by budget: {str(e)[:50]}")
                else:
                    print(f"      ❌ Failed: {str(e)[:50]}")

        print("\n📊 Budget Management Results:")
        print(f"   Successful searches: {successful_searches}")
        print(f"   Budget-blocked searches: {budget_blocked_searches}")
        print(f"   Final budget utilization: {adapter.get_cost_summary()['daily_budget_utilization']:.1f}%")

    # Demonstrate budget policy adjustments
    demonstrate_budget_policies()


def demonstrate_budget_policies():
    """Demonstrate different budget policy behaviors."""
    print("\n⚙️ Budget Policy Options:")

    policies = [
        {
            'name': 'advisory',
            'description': 'Warns about budget but allows operations',
            'behavior': 'Operations continue with cost warnings'
        },
        {
            'name': 'enforced',
            'description': 'Blocks operations that exceed budget',
            'behavior': 'Operations blocked when budget exceeded'
        },
        {
            'name': 'strict',
            'description': 'Maximum governance with pre-checks',
            'behavior': 'Operations blocked with strict validation'
        }
    ]

    for policy in policies:
        print(f"\n   🛡️ {policy['name'].upper()} Policy:")
        print(f"      Description: {policy['description']}")
        print(f"      Behavior: {policy['behavior']}")

    print("\n💡 Policy Selection Guidelines:")
    print("   • Use ADVISORY for development and testing")
    print("   • Use ENFORCED for production cost control")
    print("   • Use STRICT for maximum governance and compliance")


def demonstrate_volume_analysis(adapter, calculator):
    """Demonstrate volume pricing analysis and optimization."""
    print("\n📈 Volume Pricing Analysis")
    print("=" * 30)

    # Analyze different volume scenarios
    volume_scenarios = [
        {'name': 'Light Usage', 'daily_queries': 50, 'monthly_queries': 1500},
        {'name': 'Medium Usage', 'daily_queries': 200, 'monthly_queries': 6000},
        {'name': 'Heavy Usage', 'daily_queries': 1000, 'monthly_queries': 30000},
        {'name': 'Enterprise Usage', 'daily_queries': 5000, 'monthly_queries': 150000}
    ]

    print("💰 Volume Cost Analysis:")

    for scenario in volume_scenarios:
        analysis = adapter.get_search_cost_analysis(
            projected_queries=scenario['monthly_queries'],
            model="sonar",
            average_tokens_per_query=500
        )

        monthly_cost = analysis['current_cost_structure']['projected_total_cost']
        cost_per_query = analysis['current_cost_structure']['cost_per_query']

        print(f"\n   📊 {scenario['name']}:")
        print(f"      Daily queries: {scenario['daily_queries']}")
        print(f"      Monthly cost: ${monthly_cost:.2f}")
        print(f"      Cost per query: ${cost_per_query:.6f}")

        # Show optimization opportunities
        if analysis['optimization_opportunities']:
            top_opt = analysis['optimization_opportunities'][0]
            print(f"      💡 Optimization: {top_opt['optimization_type']}")
            print(f"      💰 Potential savings: ${top_opt['potential_savings_total']:.2f}/month")

    # Volume optimization recommendations
    print("\n🎯 Volume Optimization Strategies:")
    print("   • Light usage: Focus on query optimization and caching")
    print("   • Medium usage: Consider batch processing and model selection")
    print("   • Heavy usage: Implement intelligent routing and sampling")
    print("   • Enterprise: Custom optimization with dedicated support")


def demonstrate_cost_forecasting(calculator):
    """Demonstrate cost forecasting capabilities."""
    print("\n🔮 Cost Forecasting and Planning")
    print("=" * 35)

    # Forecast different growth scenarios
    current_usage = 100  # queries per day
    growth_scenarios = [1.2, 1.5, 2.0, 3.0]  # 20%, 50%, 100%, 200% growth

    print(f"📊 Growth Scenario Analysis (Current: {current_usage} queries/day):")

    for growth_factor in growth_scenarios:
        new_usage = int(current_usage * growth_factor)
        growth_percent = int((growth_factor - 1) * 100)

        # Calculate costs for different models
        sonar_cost = calculator.calculate_search_cost("sonar", 500, SearchContext.MEDIUM)
        sonar_pro_cost = calculator.calculate_search_cost("sonar-pro", 500, SearchContext.MEDIUM)

        monthly_sonar = float(sonar_cost * new_usage * 30)
        monthly_sonar_pro = float(sonar_pro_cost * new_usage * 30)

        print(f"\n   📈 +{growth_percent}% Growth ({new_usage} queries/day):")
        print(f"      Sonar model: ${monthly_sonar:.2f}/month")
        print(f"      Sonar Pro: ${monthly_sonar_pro:.2f}/month")
        print(f"      Cost difference: ${monthly_sonar_pro - monthly_sonar:.2f}/month")

    # Annual forecasting
    print("\n📅 Annual Cost Projections:")

    annual_scenarios = [
        {'name': 'Conservative', 'daily_avg': 150, 'model_mix': {'sonar': 0.8, 'sonar-pro': 0.2}},
        {'name': 'Moderate', 'daily_avg': 300, 'model_mix': {'sonar': 0.6, 'sonar-pro': 0.4}},
        {'name': 'Aggressive', 'daily_avg': 600, 'model_mix': {'sonar': 0.4, 'sonar-pro': 0.6}}
    ]

    for scenario in annual_scenarios:
        sonar_queries = int(scenario['daily_avg'] * scenario['model_mix']['sonar'] * 365)
        sonar_pro_queries = int(scenario['daily_avg'] * scenario['model_mix']['sonar-pro'] * 365)

        sonar_annual_cost = float(calculator.calculate_search_cost("sonar", 500, SearchContext.MEDIUM) * sonar_queries)
        sonar_pro_annual_cost = float(calculator.calculate_search_cost("sonar-pro", 500, SearchContext.MEDIUM) * sonar_pro_queries)

        total_annual_cost = sonar_annual_cost + sonar_pro_annual_cost

        print(f"\n   📈 {scenario['name']} Scenario:")
        print(f"      Daily queries: {scenario['daily_avg']}")
        print(f"      Annual cost: ${total_annual_cost:.2f}")
        print(f"      Monthly average: ${total_annual_cost / 12:.2f}")


def show_optimization_summary(adapter):
    """Show comprehensive optimization summary and recommendations."""
    print("\n🏆 Cost Optimization Summary")
    print("=" * 35)

    # Current status
    cost_summary = adapter.get_cost_summary()

    print("📊 Current Optimization Status:")
    print(f"   Daily spend: ${cost_summary['daily_costs']:.6f}")
    print(f"   Budget efficiency: {cost_summary['daily_budget_utilization']:.1f}%")
    print(f"   Governance level: {cost_summary['governance_policy']}")
    print(f"   Cost alerts: {'✅' if cost_summary.get('cost_alerts_enabled') else '❌'}")

    # Key optimization strategies
    print("\n🎯 Key Optimization Strategies:")
    print("   1. Model Selection:")
    print("      • Use 'sonar' for general queries (cost-effective)")
    print("      • Use 'sonar-pro' for research requiring citations")
    print("      • Reserve reasoning models for complex analysis")

    print("\n   2. Search Context Optimization:")
    print("      • LOW context: Simple fact-finding (lowest cost)")
    print("      • MEDIUM context: Balanced approach (recommended)")
    print("      • HIGH context: Comprehensive research (higher cost)")

    print("\n   3. Budget Management:")
    print("      • Set realistic daily/monthly limits")
    print("      • Use 'enforced' policy for cost control")
    print("      • Monitor utilization regularly")

    print("\n   4. Volume Optimization:")
    print("      • Implement query batching for efficiency")
    print("      • Use caching for repeated queries")
    print("      • Consider query sampling for high volumes")

    print("\n💡 Immediate Action Items:")

    # Generate personalized recommendations
    recommendations = []

    if cost_summary['daily_budget_utilization'] > 80:
        recommendations.append("Review budget limits - currently at high utilization")

    if cost_summary['governance_policy'] == 'advisory':
        recommendations.append("Consider 'enforced' policy for better cost control")

    if cost_summary['daily_costs'] > 10:
        recommendations.append("Analyze query patterns for optimization opportunities")

    recommendations.extend([
        "Implement query result caching for repeated searches",
        "Monitor cost per query trends weekly",
        "Set up cost alerts for budget management"
    ])

    for i, rec in enumerate(recommendations[:5], 1):
        print(f"   {i}. {rec}")

    print("\n📚 Additional Resources:")
    print("   • Review production_patterns.py for scaling strategies")
    print("   • Check docs/integrations/perplexity.md for advanced optimization")
    print("   • Monitor cost trends with your observability platform")


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Example cancelled by user.")
        exit(1)
    except Exception as e:
        print(f"\n❌ Cost optimization example failed: {e}")
        exit(1)
