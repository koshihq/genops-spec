#!/usr/bin/env python3
"""
Perplexity AI Advanced Search Patterns Example

This example demonstrates advanced Perplexity AI search patterns including
complex multi-step research workflows, citation analysis, batch processing,
and sophisticated governance controls.

Usage:
    python advanced_search.py

Prerequisites:
    pip install genops[perplexity]
    export PERPLEXITY_API_KEY="pplx-your-api-key"
    export GENOPS_TEAM="your-team-name"
    export GENOPS_PROJECT="your-project-name"

Expected Output:
    - 🔬 Multi-step research workflows with session tracking
    - 📊 Citation analysis and source quality assessment
    - ⚡ Batch search processing with optimization
    - 🎯 Advanced governance with custom policies

Learning Objectives:
    - Master complex search workflows and session management
    - Analyze citations and source quality for research
    - Implement batch processing for efficiency
    - Configure advanced governance and compliance controls

Time Required: ~15 minutes
"""

import os
import time
from typing import Any


def main():
    """Run advanced Perplexity search patterns example."""
    print("🔬 Perplexity AI + GenOps Advanced Search Patterns Example")
    print("=" * 65)
    print()
    print("This example demonstrates sophisticated search workflows including")
    print("multi-step research, citation analysis, and advanced governance.")
    print()

    try:
        from genops.providers.perplexity import (
            GenOpsPerplexityAdapter,
            PerplexityModel,  # noqa: F401
            SearchContext,
        )

        print("🔧 Initializing advanced Perplexity adapter...")

        # Advanced adapter configuration
        adapter = GenOpsPerplexityAdapter(
            team=os.getenv("GENOPS_TEAM", "advanced-research-team"),
            project=os.getenv("GENOPS_PROJECT", "advanced-search-patterns"),
            environment="development",
            customer_id="research-division",
            cost_center="ai-research-lab",
            daily_budget_limit=200.0,
            monthly_budget_limit=5000.0,
            enable_governance=True,
            governance_policy="enforced",  # Strict governance for research
            default_search_context=SearchContext.HIGH,
            tags={
                "example": "advanced_search",
                "use_case": "research_workflows",
                "complexity": "high",
                "governance_level": "enterprise",
            },
        )

        print("✅ Advanced adapter configured")
        print(f"   Team: {adapter.team} | Project: {adapter.project}")
        print(
            f"   Customer: {adapter.customer_id} | Cost Center: {adapter.cost_center}"
        )
        print(
            f"   Governance: {adapter.governance_policy} | Budget: ${adapter.daily_budget_limit}/day"
        )

        # Run advanced examples
        demonstrate_multi_step_research(adapter)
        demonstrate_citation_analysis(adapter)
        demonstrate_batch_processing(adapter)
        demonstrate_domain_filtering(adapter)

        # Show comprehensive analytics
        show_advanced_analytics(adapter)

        print("\n🎉 Advanced search patterns example completed!")
        return True

    except ImportError as e:
        print(f"❌ GenOps Perplexity provider not available: {e}")
        print("   Fix: pip install genops[perplexity]")
        return False

    except Exception as e:
        print(f"❌ Advanced example failed: {e}")
        return False


def demonstrate_multi_step_research(adapter):
    """Demonstrate complex multi-step research workflows."""
    print("\n🔬 Multi-Step Research Workflow")
    print("=" * 40)
    print("Conducting comprehensive research on 'Sustainable AI Computing'")

    # Define research workflow steps
    research_steps = [
        {
            "step": "background_research",
            "query": "What is sustainable AI computing and why is it important?",
            "model": PerplexityModel.SONAR_PRO,  # noqa: F821
            "context": SearchContext.HIGH,  # noqa: F821
            "description": "Background and context research",
        },
        {
            "step": "current_challenges",
            "query": "What are the main challenges in sustainable AI computing 2024?",
            "model": PerplexityModel.SONAR_PRO,  # noqa: F821
            "context": SearchContext.HIGH,  # noqa: F821
            "description": "Current challenges analysis",
        },
        {
            "step": "solutions_and_innovations",
            "query": "Latest innovations and solutions for energy-efficient AI systems",
            "model": PerplexityModel.SONAR_PRO,  # noqa: F821
            "context": SearchContext.HIGH,  # noqa: F821
            "description": "Solutions and innovations research",
        },
        {
            "step": "industry_adoption",
            "query": "Which companies are leading sustainable AI computing adoption?",
            "model": PerplexityModel.SONAR,  # noqa: F821
            "context": SearchContext.MEDIUM,  # noqa: F821
            "description": "Industry adoption analysis",
        },
        {
            "step": "future_trends",
            "query": "Future trends and predictions for sustainable AI computing",
            "model": PerplexityModel.SONAR_PRO,  # noqa: F821
            "context": SearchContext.MEDIUM,  # noqa: F821
            "description": "Future trends research",
        },
    ]

    research_results = {}

    with adapter.track_search_session("sustainable_ai_research") as session:
        print(f"\n📋 Research Session: {session.session_name} ({session.session_id})")

        for i, step in enumerate(research_steps, 1):
            print(f"\n   📑 Step {i}/{len(research_steps)}: {step['description']}")
            print(f'      Query: "{step["query"][:60]}..."')
            print(
                f"      Model: {step['model'].value} | Context: {step['context'].value}"
            )

            try:
                start_time = time.time()

                result = adapter.search_with_governance(
                    query=step["query"],
                    model=step["model"],
                    search_context=step["context"],
                    session_id=session.session_id,
                    max_tokens=400,
                    return_citations=True,
                    research_step=step["step"],
                    research_workflow="sustainable_ai_computing",
                )

                step_time = time.time() - start_time

                # Store results for analysis
                research_results[step["step"]] = {
                    "result": result,
                    "step_info": step,
                    "execution_time": step_time,
                }

                print(f"      ✅ Completed in {step_time:.2f}s")
                print(
                    f"      📊 {result.tokens_used} tokens | {len(result.citations)} sources | ${result.cost:.6f}"
                )

                # Brief analysis of citations
                if result.citations:
                    domains = set()
                    for citation in result.citations[:3]:  # Top 3 citations
                        if "url" in citation:
                            try:
                                domain = citation["url"].split("//")[1].split("/")[0]
                                domains.add(domain)
                            except Exception:
                                pass
                    if domains:
                        print(
                            f"      🔗 Top sources: {', '.join(list(domains)[:2])}..."
                        )

                # Small delay between steps
                time.sleep(1.5)

            except Exception as e:
                print(f"      ❌ Step failed: {str(e)[:60]}")
                continue

        # Research workflow summary
        print("\n📊 Research Workflow Summary:")
        print(f"   Total Steps: {len(research_results)}/{len(research_steps)}")
        print(f"   Session Cost: ${session.total_cost:.6f}")
        print(
            f"   Average Cost per Step: ${session.total_cost / len(research_results):.6f}"
        )
        print(
            f"   Total Citations: {sum(len(r['result'].citations) for r in research_results.values())}"
        )

        # Identify most expensive step
        if research_results:
            most_expensive = max(
                research_results.items(), key=lambda x: x[1]["result"].cost
            )
            print(
                f"   Most Expensive Step: {most_expensive[0]} (${most_expensive[1]['result'].cost:.6f})"
            )


def demonstrate_citation_analysis(adapter):
    """Demonstrate advanced citation analysis and source quality assessment."""
    print("\n📚 Citation Analysis and Source Quality Assessment")
    print("=" * 55)

    # Research query for citation analysis
    query = "Impact of large language models on software development productivity"

    with adapter.track_search_session("citation_analysis") as session:
        try:
            result = adapter.search_with_governance(
                query=query,
                model=PerplexityModel.SONAR_PRO,  # noqa: F821
                search_context=SearchContext.HIGH,  # noqa: F821
                session_id=session.session_id,
                max_tokens=500,
                return_citations=True,
                analysis_type="citation_quality",
            )

            print(f'🔍 Query: "{query}"')
            print(f"📄 Response length: {len(result.response)} characters")
            print(f"📚 Citations found: {len(result.citations)}")

            if result.citations:
                print("\n📊 Citation Quality Analysis:")

                # Analyze citation domains
                domain_analysis = analyze_citation_domains(result.citations)
                print(f"   Academic sources: {domain_analysis['academic']} citations")
                print(f"   News sources: {domain_analysis['news']} citations")
                print(f"   Technical sources: {domain_analysis['technical']} citations")
                print(f"   Other sources: {domain_analysis['other']} citations")

                # Show top citations with analysis
                print("\n🏆 Top Citations Analysis:")
                for i, citation in enumerate(result.citations[:3], 1):
                    domain_type = classify_source_domain(citation.get("url", ""))
                    title = citation.get("title", "No title")[:60]

                    print(f"   {i}. {title}...")
                    print(f"      URL: {citation.get('url', 'N/A')[:70]}...")
                    print(f"      Type: {domain_type}")
                    print(f"      Snippet: {citation.get('snippet', 'N/A')[:80]}...")
                    print()
            else:
                print(
                    "   ⚠️ No citations found - this may indicate a general knowledge query"
                )

        except Exception as e:
            print(f"❌ Citation analysis failed: {e}")


def demonstrate_batch_processing(adapter):
    """Demonstrate efficient batch search processing."""
    print("\n⚡ Batch Search Processing")
    print("=" * 30)
    print("Processing multiple related queries efficiently...")

    # Define a set of related queries for batch processing
    batch_queries = [
        "Best practices for microservices architecture",
        "Container orchestration with Kubernetes best practices",
        "Monitoring and observability in distributed systems",
        "Security considerations for cloud-native applications",
        "DevOps automation tools and workflows",
    ]

    print(f"📋 Processing {len(batch_queries)} related queries...")

    try:
        start_time = time.time()

        # Use batch processing
        results = adapter.batch_search_with_governance(
            queries=batch_queries,
            model=PerplexityModel.SONAR,  # noqa: F821
            search_context=SearchContext.MEDIUM,  # noqa: F821
            batch_optimization=True,
            research_topic="cloud_native_development",
        )

        total_time = time.time() - start_time

        print("\n📊 Batch Processing Results:")
        print(f"   Queries processed: {len(results)}/{len(batch_queries)}")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Average time per query: {total_time / len(results):.2f} seconds")

        # Cost analysis
        total_cost = sum(result.cost for result in results)
        total_tokens = sum(result.tokens_used for result in results)

        print(f"   Total cost: ${total_cost:.6f}")
        print(f"   Average cost per query: ${total_cost / len(results):.6f}")
        print(f"   Total tokens: {total_tokens}")
        print(f"   Cost efficiency: ${total_cost / total_tokens:.8f} per token")

        # Show sample results
        print("\n🔍 Sample Results:")
        for i, (query, result) in enumerate(zip(batch_queries[:2], results[:2])):
            print(f"   Query {i + 1}: {query[:50]}...")
            print(f"   Response: {result.response[:100]}...")
            print(f"   Cost: ${result.cost:.6f} | Citations: {len(result.citations)}")
            print()

    except Exception as e:
        print(f"❌ Batch processing failed: {e}")


def demonstrate_domain_filtering(adapter):
    """Demonstrate domain filtering and source control."""
    print("\n🎯 Domain Filtering and Source Control")
    print("=" * 45)
    print("Controlling search sources for quality and relevance...")

    # Different domain filtering scenarios
    filtering_scenarios = [
        {
            "name": "Academic Sources Only",
            "query": "Machine learning interpretability methods",
            "filter": ["arxiv.org", "scholar.google.com", "ieee.org", "acm.org"],
            "description": "Academic and research sources",
        },
        {
            "name": "News and Industry",
            "query": "Latest AI industry developments",
            "filter": ["techcrunch.com", "venturebeat.com", "wired.com", "reuters.com"],
            "description": "News and industry publications",
        },
        {
            "name": "Technical Documentation",
            "query": "Python machine learning library comparison",
            "filter": [
                "docs.python.org",
                "scikit-learn.org",
                "pytorch.org",
                "tensorflow.org",
            ],
            "description": "Official documentation sources",
        },
    ]

    with adapter.track_search_session("domain_filtering_demo") as session:
        for scenario in filtering_scenarios:
            print(f"\n📂 {scenario['name']}:")
            print(f"   Description: {scenario['description']}")
            print(f"   Allowed domains: {', '.join(scenario['filter'][:2])}...")

            try:
                result = adapter.search_with_governance(
                    query=scenario["query"],
                    model=PerplexityModel.SONAR,  # noqa: F821
                    search_context=SearchContext.MEDIUM,  # noqa: F821
                    session_id=session.session_id,
                    search_domain_filter=scenario["filter"],
                    max_tokens=200,
                )

                print(
                    f"   ✅ Search completed: {len(result.citations)} citations found"
                )

                # Verify domain filtering worked
                if result.citations:
                    filtered_domains = []
                    for citation in result.citations:
                        if "url" in citation:
                            try:
                                domain = citation["url"].split("//")[1].split("/")[0]
                                filtered_domains.append(domain)
                            except Exception:
                                pass

                    print(
                        f"   📊 Result domains: {', '.join(set(filtered_domains)[:3])}..."
                    )

                    # Check if filtering was effective
                    allowed_domains = set(scenario["filter"])
                    found_domains = set(filtered_domains)

                    if any(domain in allowed_domains for domain in found_domains):
                        print("   ✅ Domain filtering effective")
                    else:
                        print("   ⚠️ Domain filtering may not have been applied")

                print(f"   💰 Cost: ${result.cost:.6f}")

            except Exception as e:
                print(f"   ❌ Filtering scenario failed: {str(e)[:50]}")


def analyze_citation_domains(citations: list[dict[str, Any]]) -> dict[str, int]:
    """Analyze citation domains to categorize source types."""
    domain_counts = {"academic": 0, "news": 0, "technical": 0, "other": 0}

    academic_domains = {
        "arxiv.org",
        "scholar.google.com",
        "ieee.org",
        "acm.org",
        "springer.com",
        "nature.com",
    }
    news_domains = {
        "bbc.com",
        "cnn.com",
        "reuters.com",
        "techcrunch.com",
        "wired.com",
        "venturebeat.com",
    }
    technical_domains = {
        "github.com",
        "stackoverflow.com",
        "docs.python.org",
        "medium.com",
    }

    for citation in citations:
        url = citation.get("url", "")
        if url:
            try:
                domain = url.split("//")[1].split("/")[0].lower()

                if any(d in domain for d in academic_domains):
                    domain_counts["academic"] += 1
                elif any(d in domain for d in news_domains):
                    domain_counts["news"] += 1
                elif any(d in domain for d in technical_domains):
                    domain_counts["technical"] += 1
                else:
                    domain_counts["other"] += 1
            except Exception:
                domain_counts["other"] += 1

    return domain_counts


def classify_source_domain(url: str) -> str:
    """Classify a source URL by domain type."""
    if not url:
        return "unknown"

    try:
        domain = url.split("//")[1].split("/")[0].lower()

        if any(
            d in domain
            for d in ["arxiv", "scholar", "ieee", "acm", "springer", "nature"]
        ):
            return "academic"
        elif any(
            d in domain
            for d in ["bbc", "cnn", "reuters", "techcrunch", "wired", "venturebeat"]
        ):
            return "news"
        elif any(d in domain for d in ["github", "stackoverflow", "docs", "medium"]):
            return "technical"
        elif any(d in domain for d in [".gov", ".edu"]):
            return "institutional"
        else:
            return "other"
    except Exception:
        return "unknown"


def show_advanced_analytics(adapter):
    """Display advanced analytics and insights."""
    print("\n📊 Advanced Analytics and Insights")
    print("=" * 40)

    # Comprehensive cost analysis
    cost_summary = adapter.get_cost_summary()

    print("💰 Cost Intelligence:")
    print(f"   Total Daily Spend: ${cost_summary['daily_costs']:.6f}")
    print(f"   Budget Utilization: {cost_summary['daily_budget_utilization']:.1f}%")
    print(f"   Active Sessions: {cost_summary['active_sessions']}")

    # Advanced cost analysis for high-volume scenarios
    try:
        analysis = adapter.get_search_cost_analysis(
            projected_queries=1000, model="sonar-pro", average_tokens_per_query=400
        )

        print("\n🎯 High-Volume Cost Projections (1000 searches):")
        print(
            f"   Total Projected Cost: ${analysis['current_cost_structure']['projected_total_cost']:.4f}"
        )
        print(
            f"   Cost per Search: ${analysis['current_cost_structure']['cost_per_query']:.6f}"
        )

        if analysis["optimization_opportunities"]:
            print("\n💡 Top 3 Optimization Opportunities:")
            for i, opt in enumerate(analysis["optimization_opportunities"][:3], 1):
                print(
                    f"   {i}. {opt['optimization_type']}: ${opt['potential_savings_total']:.4f} savings"
                )
                print(f"      Description: {opt['description']}")

        if analysis["recommendations"]:
            print("\n📋 Recommendations:")
            for rec in analysis["recommendations"][:3]:
                print(f"   • {rec}")

    except Exception as e:
        print(f"   Advanced analysis unavailable: {str(e)[:50]}")

    print("\n🏆 Advanced Pattern Benefits:")
    print("   ✅ Multi-step research workflows")
    print("   ✅ Citation quality analysis")
    print("   ✅ Batch processing optimization")
    print("   ✅ Domain filtering and source control")
    print("   ✅ Comprehensive cost intelligence")


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Example cancelled by user.")
        exit(1)
    except Exception as e:
        print(f"\n❌ Advanced example failed: {e}")
        exit(1)
