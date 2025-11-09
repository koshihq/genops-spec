#!/usr/bin/env python3
"""
Perplexity AI Basic Real-Time Search Example

This example demonstrates basic Perplexity AI real-time web search with GenOps
governance, including cost attribution, team management, and citation tracking.

Usage:
    python basic_search.py

Prerequisites:
    pip install genops[perplexity]
    export PERPLEXITY_API_KEY="pplx-your-api-key"
    export GENOPS_TEAM="your-team-name"
    export GENOPS_PROJECT="your-project-name"

Expected Output:
    - ✅ Real-time web search results with citations
    - 💰 Detailed cost breakdown (token + request costs)
    - 🏷️ Team and project attribution tracking
    - 📊 Search performance metrics and optimization tips

Learning Objectives:
    - Perform real-time web searches with Perplexity AI
    - Understand Perplexity's dual pricing model (tokens + requests)
    - Learn citation tracking and source attribution
    - Practice basic governance with cost tracking

Time Required: ~5 minutes
"""

import os
import time
import random
from datetime import datetime, timezone
from decimal import Decimal


def main():
    """Run basic Perplexity search example with GenOps governance."""
    print("🔍 Perplexity AI + GenOps Basic Real-Time Search Example")
    print("=" * 65)
    print()
    print("This example demonstrates real-time web search with Perplexity AI,")
    print("including cost tracking, citation management, and governance controls.")
    print()

    # Prerequisites check
    print("📋 Prerequisites Check:")
    prerequisites = [
        ("GenOps installed", "genops"),
        ("OpenAI client available", "openai"),
        ("PERPLEXITY_API_KEY configured", lambda: bool(os.getenv('PERPLEXITY_API_KEY'))),
        ("GENOPS_TEAM configured", lambda: bool(os.getenv('GENOPS_TEAM')))
    ]
    
    for desc, check in prerequisites:
        try:
            if callable(check):
                result = check()
            else:
                __import__(check)
                result = True
            print(f"  ✅ {desc}")
        except (ImportError, Exception):
            print(f"  ❌ {desc}")
            if desc.startswith("GenOps"):
                print("     Fix: pip install genops[perplexity]")
            elif "API_KEY" in desc:
                print("     Fix: export PERPLEXITY_API_KEY='pplx-your-api-key'")
                print("     Get key: https://www.perplexity.ai/settings/api")
            elif "TEAM" in desc:
                print("     Optional: export GENOPS_TEAM='your-team-name'")

    try:
        from genops.providers.perplexity import GenOpsPerplexityAdapter, PerplexityModel, SearchContext
        
        print("\n🔧 Initializing Perplexity adapter with governance...")
        
        # Create adapter with governance configuration
        adapter = GenOpsPerplexityAdapter(
            team=os.getenv('GENOPS_TEAM', 'search-demo-team'),
            project=os.getenv('GENOPS_PROJECT', 'basic-search-example'),
            environment='development',
            daily_budget_limit=50.0,  # Conservative limit for demo
            enable_governance=True,
            governance_policy='advisory',  # Allow operations with warnings
            tags={
                'example': 'basic_search',
                'use_case': 'real_time_research',
                'demo_mode': 'true'
            }
        )
        
        print("✅ Adapter configured with governance enabled")
        print(f"   Team: {adapter.team}")
        print(f"   Project: {adapter.project}")
        print(f"   Daily Budget: ${adapter.daily_budget_limit}")
        print(f"   Governance: {adapter.governance_policy}")

        # Demonstrate basic search scenarios
        demonstrate_basic_search(adapter)
        demonstrate_search_contexts(adapter)
        demonstrate_model_comparison(adapter)
        
        # Show cost summary
        show_cost_summary(adapter)
        
        print("\n🎉 Basic search example completed successfully!")
        return True
        
    except ImportError as e:
        print(f"\n❌ GenOps Perplexity provider not available: {e}")
        print("   Fix: pip install genops[perplexity]")
        return False
    
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        return False


def demonstrate_basic_search(adapter):
    """Demonstrate basic search with different query types."""
    print("\n🌐 Basic Real-Time Search Demonstrations")
    print("=" * 50)
    
    # Example searches of different types
    search_examples = [
        {
            'query': 'Latest developments in artificial intelligence 2024',
            'description': 'Current news and trends',
            'model': PerplexityModel.SONAR,
            'context': SearchContext.MEDIUM
        },
        {
            'query': 'Best practices for Python error handling',
            'description': 'Technical documentation search',
            'model': PerplexityModel.SONAR,
            'context': SearchContext.LOW
        },
        {
            'query': 'Climate change impact on renewable energy adoption',
            'description': 'Academic research topic',
            'model': PerplexityModel.SONAR_PRO,
            'context': SearchContext.HIGH
        }
    ]
    
    with adapter.track_search_session("basic_search_demo") as session:
        for i, example in enumerate(search_examples, 1):
            print(f"\n📱 Search Example {i}: {example['description']}")
            print(f"   Query: \"{example['query']}\"")
            print(f"   Model: {example['model'].value}")
            print(f"   Context: {example['context'].value}")
            
            try:
                start_time = time.time()
                
                result = adapter.search_with_governance(
                    query=example['query'],
                    model=example['model'],
                    search_context=example['context'],
                    session_id=session.session_id,
                    max_tokens=300,  # Limit for demo
                    return_citations=True,
                    search_query_type=example['description'].lower().replace(' ', '_')
                )
                
                search_time = time.time() - start_time
                
                # Display results
                print(f"\n   📄 Search Results:")
                response_preview = result.response[:200] + "..." if len(result.response) > 200 else result.response
                print(f"      Response: {response_preview}")
                print(f"      Citations: {len(result.citations)} sources found")
                
                # Show first citation as example
                if result.citations:
                    citation = result.citations[0]
                    print(f"      Example Citation: {citation.get('title', 'N/A')[:50]}...")
                    print(f"                        {citation.get('url', 'N/A')[:60]}...")
                
                # Cost and performance metrics
                print(f"\n   💰 Cost Analysis:")
                print(f"      Tokens Used: {result.tokens_used}")
                print(f"      Total Cost: ${result.cost:.6f}")
                print(f"      Cost per Token: ${(result.cost / result.tokens_used):.8f}")
                print(f"      Search Time: {search_time:.2f} seconds")
                
                # Brief delay between searches
                time.sleep(1)
                
            except Exception as e:
                print(f"   ❌ Search failed: {str(e)[:100]}")
                continue
        
        print(f"\n📊 Session Summary:")
        print(f"   Total Searches: {session.total_queries}")
        print(f"   Total Cost: ${session.total_cost:.6f}")
        print(f"   Average Cost per Search: ${(session.total_cost / session.total_queries):.6f}")


def demonstrate_search_contexts(adapter):
    """Demonstrate different search context depths and their cost impact."""
    print("\n📊 Search Context Comparison")
    print("=" * 40)
    print("Search contexts affect request costs and result depth:")
    print("• LOW: Basic search, lower cost, faster")
    print("• MEDIUM: Balanced search depth and cost")
    print("• HIGH: Comprehensive search, higher cost")
    
    query = "Machine learning best practices for production systems"
    contexts = [SearchContext.LOW, SearchContext.MEDIUM, SearchContext.HIGH]
    
    context_results = []
    
    with adapter.track_search_session("context_comparison") as session:
        for context in contexts:
            print(f"\n🔍 Testing {context.value.upper()} context:")
            
            try:
                result = adapter.search_with_governance(
                    query=query,
                    model=PerplexityModel.SONAR,
                    search_context=context,
                    session_id=session.session_id,
                    max_tokens=200
                )
                
                context_results.append({
                    'context': context.value,
                    'cost': result.cost,
                    'tokens': result.tokens_used,
                    'citations': len(result.citations),
                    'search_time': result.search_time_seconds
                })
                
                print(f"   Cost: ${result.cost:.6f}")
                print(f"   Citations: {len(result.citations)}")
                print(f"   Time: {result.search_time_seconds:.2f}s")
                
            except Exception as e:
                print(f"   ❌ Failed: {str(e)[:50]}")
    
    # Context comparison summary
    if len(context_results) > 1:
        print(f"\n📈 Context Impact Analysis:")
        low_cost = next((r['cost'] for r in context_results if r['context'] == 'low'), None)
        high_cost = next((r['cost'] for r in context_results if r['context'] == 'high'), None)
        
        if low_cost and high_cost:
            cost_increase = ((high_cost / low_cost - 1) * 100)
            print(f"   Cost increase from LOW to HIGH: {cost_increase:.1f}%")
            print(f"   Recommendation: Use MEDIUM context for balanced cost/quality")


def demonstrate_model_comparison(adapter):
    """Demonstrate different Perplexity models and their capabilities."""
    print("\n🤖 Model Comparison")
    print("=" * 25)
    
    query = "Explain quantum computing applications"
    models = [PerplexityModel.SONAR, PerplexityModel.SONAR_PRO]
    
    with adapter.track_search_session("model_comparison") as session:
        for model in models:
            print(f"\n🧠 Testing {model.value.upper()} model:")
            
            try:
                result = adapter.search_with_governance(
                    query=query,
                    model=model,
                    search_context=SearchContext.MEDIUM,
                    session_id=session.session_id,
                    max_tokens=150
                )
                
                print(f"   Response length: {len(result.response)} chars")
                print(f"   Citations found: {len(result.citations)}")
                print(f"   Cost: ${result.cost:.6f}")
                print(f"   Cost per token: ${(result.cost / result.tokens_used):.8f}")
                
            except Exception as e:
                print(f"   ❌ Model test failed: {str(e)[:50]}")


def show_cost_summary(adapter):
    """Display comprehensive cost summary and recommendations."""
    print("\n💰 Cost Intelligence Summary")
    print("=" * 35)
    
    summary = adapter.get_cost_summary()
    
    print(f"📊 Current Usage:")
    print(f"   Daily Costs: ${summary['daily_costs']:.6f}")
    print(f"   Budget Utilization: {summary['daily_budget_utilization']:.1f}%")
    print(f"   Remaining Budget: ${summary['daily_budget_limit'] - summary['daily_costs']:.4f}")
    
    # Cost optimization analysis
    try:
        analysis = adapter.get_search_cost_analysis(
            projected_queries=100,
            model="sonar"
        )
        
        print(f"\n🎯 Cost Projections (100 searches):")
        print(f"   Estimated Total: ${analysis['current_cost_structure']['projected_total_cost']:.4f}")
        print(f"   Cost per Search: ${analysis['current_cost_structure']['cost_per_query']:.6f}")
        
        if analysis['optimization_opportunities']:
            top_optimization = analysis['optimization_opportunities'][0]
            print(f"\n💡 Top Optimization Opportunity:")
            print(f"   {top_optimization['description']}")
            print(f"   Potential Savings: ${top_optimization['potential_savings_total']:.4f}")
        
    except Exception as e:
        print(f"   Note: Advanced cost analysis unavailable: {str(e)[:50]}")
    
    print(f"\n📈 Optimization Tips:")
    print(f"   • Use 'sonar' model for cost-effective searches")
    print(f"   • Choose 'low' context for simple queries")
    print(f"   • Batch similar searches to reduce request fees")
    print(f"   • Monitor budget utilization with daily limits")


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Example cancelled by user.")
        exit(1)
    except Exception as e:
        print(f"\n❌ Example failed with unexpected error: {e}")
        exit(1)