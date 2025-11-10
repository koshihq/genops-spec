#!/usr/bin/env python3
"""
📊 GenOps LlamaIndex RAG Pipeline Tracking - Phase 2 (20 minutes)

This example demonstrates comprehensive RAG pipeline monitoring with GenOps.
Shows detailed cost breakdown, quality metrics, and performance optimization.

What you'll learn:
- Complete RAG component tracking (embeddings, retrieval, synthesis)
- Quality metrics (retrieval relevance, content diversity) 
- Performance profiling and optimization suggestions
- Team-based cost attribution and budgeting
- Multi-provider cost comparison

Requirements:
- API key: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY
- pip install llama-index genops-ai

Usage:
    python rag_pipeline_tracking.py
"""

import os
import time


def setup_llm_provider():
    """Configure LLM provider and return provider info."""
    from llama_index.core import Settings

    provider_info = {}

    if os.getenv("OPENAI_API_KEY"):
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI
        Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
        provider_info = {
            "name": "OpenAI",
            "llm_model": "gpt-3.5-turbo",
            "embedding_model": "text-embedding-ada-002",
            "estimated_cost_per_query": "$0.002-0.01"
        }
    elif os.getenv("ANTHROPIC_API_KEY"):
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.anthropic import Anthropic
        Settings.llm = Anthropic(model="claude-3-haiku-20240307")
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        provider_info = {
            "name": "Anthropic + HuggingFace",
            "llm_model": "claude-3-haiku",
            "embedding_model": "all-MiniLM-L6-v2",
            "estimated_cost_per_query": "$0.001-0.005"
        }
    elif os.getenv("GOOGLE_API_KEY"):
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.gemini import Gemini
        Settings.llm = Gemini(model="gemini-pro")
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        provider_info = {
            "name": "Google Gemini + HuggingFace",
            "llm_model": "gemini-pro",
            "embedding_model": "all-MiniLM-L6-v2",
            "estimated_cost_per_query": "$0.0005-0.002"
        }
    else:
        raise ValueError("No API key found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")

    return provider_info

def create_knowledge_base():
    """Create a comprehensive knowledge base for RAG testing."""
    from llama_index.core import Document

    documents = [
        Document(
            text="""
            RAG System Architecture and Components
            
            A RAG (Retrieval-Augmented Generation) system consists of several key components:
            
            1. Document Ingestion: Process and chunk documents into manageable pieces
            2. Embedding Generation: Convert text chunks into vector representations
            3. Vector Storage: Store embeddings in a searchable vector database
            4. Query Processing: Convert user queries into embedding vectors
            5. Retrieval: Find most relevant document chunks using similarity search
            6. Context Assembly: Combine retrieved chunks into coherent context
            7. Response Generation: Use LLM to generate answer from retrieved context
            
            Each component has different cost and performance characteristics that need monitoring.
            """,
            metadata={"category": "architecture", "complexity": "high", "tokens": 150}
        ),

        Document(
            text="""
            Cost Optimization Strategies for RAG Systems
            
            Embedding Optimization:
            - Use smaller embedding models for simple content (384d vs 1536d)
            - Cache embeddings to avoid recomputation
            - Batch embedding requests to reduce API overhead
            
            Retrieval Optimization:
            - Tune similarity thresholds (0.7-0.8 typical)
            - Use hybrid search (keyword + semantic) for better relevance
            - Implement re-ranking for improved precision
            
            Generation Optimization:
            - Use cheaper models for simple questions (gpt-3.5 vs gpt-4)
            - Implement response caching for common queries
            - Optimize prompt templates to reduce token usage
            
            Typical cost breakdown: 60% generation, 25% embeddings, 15% infrastructure
            """,
            metadata={"category": "optimization", "complexity": "medium", "tokens": 180}
        ),

        Document(
            text="""
            RAG Quality Metrics and Evaluation
            
            Retrieval Quality:
            - Precision@K: Percentage of retrieved docs that are relevant
            - Recall@K: Percentage of relevant docs that are retrieved
            - MRR (Mean Reciprocal Rank): Average inverse rank of first relevant doc
            
            Generation Quality:
            - Faithfulness: Generated response consistency with source docs
            - Answer Relevancy: How well response addresses the query
            - Context Precision: Relevance of retrieved context chunks
            
            Performance Metrics:
            - End-to-end latency (target: <3s for simple queries)
            - Individual component latency (embedding: <200ms, retrieval: <500ms)
            - Throughput (queries per second sustained)
            
            Quality-cost tradeoffs require continuous monitoring and optimization.
            """,
            metadata={"category": "evaluation", "complexity": "high", "tokens": 200}
        ),

        Document(
            text="""
            Team-Based RAG Governance Framework
            
            Research Teams:
            - Budget: $1000-5000/month depending on scale
            - Focus: High-quality results, experimentation with advanced models
            - Metrics: Answer quality, innovation potential
            
            Engineering Teams:
            - Budget: $500-2000/month for production workloads
            - Focus: Cost efficiency, reliability, latency
            - Metrics: Cost per query, system availability, response time
            
            Product Teams:
            - Budget: $200-1000/month for feature development
            - Focus: User experience, A/B testing capabilities
            - Metrics: User satisfaction, feature adoption, conversion rates
            
            Customer Success Teams:
            - Budget: $100-500/month for support automation
            - Focus: Accurate answers, quick resolution
            - Metrics: Resolution rate, customer satisfaction, deflection rate
            """,
            metadata={"category": "governance", "complexity": "medium", "tokens": 220}
        )
    ]

    return documents

def demonstrate_comprehensive_rag_monitoring():
    """Show comprehensive RAG pipeline monitoring with all GenOps features."""
    from llama_index.core import VectorStoreIndex

    from genops.providers.llamaindex import (
        create_llamaindex_cost_context,
        create_rag_monitor,
    )

    print("📊 COMPREHENSIVE RAG PIPELINE MONITORING")
    print("=" * 50)

    # Create documents
    documents = create_knowledge_base()
    print(f"✅ Created knowledge base with {len(documents)} documents")

    # Create RAG monitor for quality and performance tracking
    rag_monitor = create_rag_monitor(
        enable_quality_metrics=True,
        enable_cost_tracking=True,
        enable_performance_profiling=True
    )
    print("✅ RAG monitor configured with quality & performance tracking")

    # Create index
    print("🔍 Building vector index (monitoring embedding costs)...")
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=3)

    # Test queries representing different team use cases
    team_queries = [
        {
            "query": "What are the key components of RAG architecture?",
            "team": "engineering",
            "project": "system-design",
            "complexity": "high",
            "expected_cost": "medium"
        },
        {
            "query": "How can we optimize RAG costs?",
            "team": "product",
            "project": "cost-optimization",
            "complexity": "medium",
            "expected_cost": "low"
        },
        {
            "query": "What metrics should we track for RAG quality?",
            "team": "research",
            "project": "evaluation-framework",
            "complexity": "high",
            "expected_cost": "high"
        },
        {
            "query": "What budget should each team have for RAG?",
            "team": "customer-success",
            "project": "support-automation",
            "complexity": "low",
            "expected_cost": "low"
        }
    ]

    # Track queries with comprehensive monitoring
    with create_llamaindex_cost_context("team_rag_demo", budget_limit=2.0, enable_alerts=True) as cost_context:

        for i, query_info in enumerate(team_queries, 1):
            print(f"\n📋 Query {i}: {query_info['team']} team")
            print(f"   Question: {query_info['query']}")

            # Use RAG monitor for detailed tracking
            with rag_monitor.monitor_rag_operation(
                query_info['query'],
                team=query_info['team'],
                project=query_info['project'],
                complexity=query_info['complexity']
            ) as monitor:

                start_time = time.time()
                response = query_engine.query(query_info['query'])
                end_time = time.time()

                # Record detailed metrics (in production, this would be automatic)
                query_time_ms = (end_time - start_time) * 1000

                print(f"   🤖 Response: {response.response[:100]}...")
                print(f"   ⚡ Latency: {query_time_ms:.0f}ms")
                print(f"   🏷️  Attribution: {query_info['team']}/{query_info['project']}")

        # Get comprehensive cost summary
        print("\n" + "=" * 50)
        print("💰 COST BREAKDOWN BY COMPONENT")
        print("=" * 50)

        summary = cost_context.get_current_summary()

        print(f"Total Pipeline Cost: ${summary.total_cost:.6f}")
        print(f"Total Operations: {summary.operation_count}")
        print(f"RAG Pipelines: {summary.rag_pipelines}")

        breakdown = summary.cost_breakdown
        print("\nComponent Costs:")
        print(f"  • Embeddings: ${breakdown.embedding_cost:.6f} ({breakdown.embedding_cost/summary.total_cost*100:.1f}%)")
        print(f"  • Retrieval: ${breakdown.retrieval_cost:.6f} ({breakdown.retrieval_cost/summary.total_cost*100:.1f}%)")
        print(f"  • Synthesis: ${breakdown.synthesis_cost:.6f} ({breakdown.synthesis_cost/summary.total_cost*100:.1f}%)")

        if breakdown.cost_by_provider:
            print("\nCosts by Provider:")
            for provider, cost in breakdown.cost_by_provider.items():
                print(f"  • {provider}: ${cost:.6f}")

        if breakdown.optimization_suggestions:
            print("\n💡 Optimization Suggestions:")
            for suggestion in breakdown.optimization_suggestions:
                print(f"  • {suggestion}")

    # Get RAG analytics
    print("\n" + "=" * 50)
    print("📈 RAG PIPELINE ANALYTICS")
    print("=" * 50)

    analytics = rag_monitor.get_analytics()

    print("Pipeline Performance:")
    print(f"  • Average Cost per Query: ${analytics.avg_cost_per_query:.6f}")
    print(f"  • Average Response Time: {analytics.avg_response_time_ms:.0f}ms")
    print(f"  • Success Rates: Embedding {analytics.embedding_success_rate*100:.1f}%, Retrieval {analytics.retrieval_success_rate*100:.1f}%, Synthesis {analytics.synthesis_success_rate*100:.1f}%")

    if analytics.avg_retrieval_relevance:
        print(f"  • Average Retrieval Relevance: {analytics.avg_retrieval_relevance:.3f}")

    if analytics.recommendations:
        print("\nPipeline Recommendations:")
        for rec in analytics.recommendations:
            print(f"  • {rec}")

    return summary, analytics

def demonstrate_team_cost_attribution():
    """Show detailed team-based cost attribution and budgeting."""
    from llama_index.core import Document, VectorStoreIndex

    from genops.providers.llamaindex import (
        create_llamaindex_cost_context,
        instrument_llamaindex,
    )

    print("\n" + "=" * 50)
    print("🏷️ TEAM COST ATTRIBUTION DEMO")
    print("=" * 50)

    # Create simple knowledge base
    docs = [
        Document(text="Customer support best practices include quick response times, accurate information, and empathetic communication."),
        Document(text="Product development requires user research, iterative design, and continuous testing to ensure market fit."),
        Document(text="Engineering teams focus on code quality, system reliability, and scalable architecture design.")
    ]

    index = VectorStoreIndex.from_documents(docs)
    query_engine = index.as_query_engine()

    # Create adapter with default governance
    adapter = instrument_llamaindex()

    # Simulate different team usage patterns
    team_scenarios = [
        {
            "team": "customer-success",
            "project": "support-automation",
            "queries": [
                "What are customer support best practices?",
                "How should we handle customer complaints?"
            ],
            "budget": 0.50
        },
        {
            "team": "product",
            "project": "feature-research",
            "queries": [
                "What makes a successful product?",
                "How do we ensure product-market fit?"
            ],
            "budget": 1.00
        },
        {
            "team": "engineering",
            "project": "system-architecture",
            "queries": [
                "What are engineering best practices?",
                "How do we build scalable systems?"
            ],
            "budget": 0.75
        }
    ]

    team_costs = {}

    for scenario in team_scenarios:
        print(f"\n👥 Team: {scenario['team']}")
        print(f"   Project: {scenario['project']}")
        print(f"   Budget: ${scenario['budget']:.2f}")

        with create_llamaindex_cost_context(
            f"{scenario['team']}_queries",
            budget_limit=scenario['budget'],
            enable_alerts=True
        ) as team_context:

            for query in scenario['queries']:
                print(f"   🤖 Query: {query}")

                response = adapter.track_query(
                    query_engine,
                    query,
                    team=scenario['team'],
                    project=scenario['project'],
                    environment="demo"
                )

                print(f"   💬 Response: {response.response[:60]}...")

            team_summary = team_context.get_current_summary()
            team_costs[scenario['team']] = team_summary.total_cost

            print(f"   💰 Team Total: ${team_summary.total_cost:.6f}")
            print(f"   📊 Budget Used: {team_summary.total_cost/scenario['budget']*100:.1f}%")

            if team_summary.budget_status and team_summary.budget_status['alerts']:
                print(f"   ⚠️  Budget Alerts: {len(team_summary.budget_status['alerts'])}")

    # Summary across teams
    print("\n📊 CROSS-TEAM SUMMARY:")
    print(f"   Total Organizational Cost: ${sum(team_costs.values()):.6f}")
    for team, cost in team_costs.items():
        print(f"   {team}: ${cost:.6f}")

def main():
    """Main demonstration of comprehensive RAG pipeline tracking."""
    print("📊 GenOps LlamaIndex RAG Pipeline Tracking")
    print("=" * 60)

    try:
        # Setup
        provider_info = setup_llm_provider()
        print(f"✅ Provider: {provider_info['name']}")
        print(f"✅ LLM Model: {provider_info['llm_model']}")
        print(f"✅ Embedding Model: {provider_info['embedding_model']}")
        print(f"✅ Estimated Cost/Query: {provider_info['estimated_cost_per_query']}")

        # Demo 1: Comprehensive monitoring
        summary, analytics = demonstrate_comprehensive_rag_monitoring()

        # Demo 2: Team attribution
        demonstrate_team_cost_attribution()

        # Final summary
        print("\n" + "=" * 60)
        print("🎉 RAG PIPELINE TRACKING COMPLETE!")
        print("=" * 60)

        print("✅ WHAT YOU ACCOMPLISHED:")
        print("   • Complete RAG component tracking (embeddings, retrieval, synthesis)")
        print("   • Quality metrics monitoring (retrieval relevance, performance)")
        print("   • Team-based cost attribution and budget management")
        print("   • Optimization suggestions for cost and performance")
        print("   • Cross-team governance and spending visibility")

        print("\n🎯 KEY INSIGHTS:")
        print(f"   • Total demo cost: ${summary.total_cost:.6f}")
        print(f"   • Average query latency: {analytics.avg_response_time_ms:.0f}ms")
        print(f"   • Most expensive component: {'Synthesis' if summary.cost_breakdown.synthesis_cost > summary.cost_breakdown.embedding_cost else 'Embeddings'}")
        print("   • Team attribution enables precise cost allocation")
        print("   • Quality monitoring identifies optimization opportunities")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")

        if "api key" in str(e).lower():
            print("\n🔧 API KEY ISSUE:")
            print("   Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY")
        else:
            print("\n🔧 For detailed diagnostics run:")
            print("   python -c \"from genops.providers.llamaindex.validation import validate_setup, print_validation_result; print_validation_result(validate_setup(), detailed=True)\"")

        return False

if __name__ == "__main__":
    success = main()

    if success:
        print("\n🚀 READY FOR PHASE 3? (Advanced Features)")
        print("   → python advanced_agent_governance.py      # Agent workflows")
        print("   → python production_rag_deployment.py      # Enterprise deployment")
        print()
        print("📚 Or explore more Phase 2 examples:")
        print("   → python embedding_cost_optimization.py    # Embedding efficiency")
    else:
        print("\n💡 Need help?")
        print("   → examples/llamaindex/README.md#troubleshooting")

    exit(0 if success else 1)
