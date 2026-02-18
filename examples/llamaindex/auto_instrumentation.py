#!/usr/bin/env python3
"""
🔧 GenOps LlamaIndex Auto-Instrumentation - Phase 2 (15 minutes)

This example shows how to add GenOps cost tracking to existing LlamaIndex
applications with ZERO code changes. Perfect for retrofitting existing RAG pipelines.

What you'll learn:
- Zero-code instrumentation with auto_instrument()
- How GenOps tracks existing RAG workflows
- Team and project cost attribution
- RAG component cost breakdown

Requirements:
- API key: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY
- pip install llama-index genops-ai

Usage:
    python auto_instrumentation.py
"""

import os


def setup_llm_provider():
    """Configure LLM provider based on available API keys."""
    from llama_index.core import Settings

    if os.getenv("OPENAI_API_KEY"):
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI

        Settings.llm = OpenAI(model="gpt-3.5-turbo")
        Settings.embed_model = OpenAIEmbedding()
        return "OpenAI"
    elif os.getenv("ANTHROPIC_API_KEY"):
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.anthropic import Anthropic

        Settings.llm = Anthropic(model="claude-3-haiku-20240307")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        return "Anthropic"
    elif os.getenv("GOOGLE_API_KEY"):
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.gemini import Gemini

        Settings.llm = Gemini(model="gemini-pro")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        return "Google Gemini"
    else:
        raise ValueError(
            "No API key found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY"
        )


def create_sample_documents():
    """Create sample documents for RAG pipeline."""
    from llama_index.core import Document

    return [
        Document(
            text="""
        Artificial Intelligence (AI) Cost Management Best Practices:

        1. Monitor Usage: Track token consumption and API calls across all models
        2. Set Budgets: Establish spending limits for different teams and projects
        3. Optimize Models: Use appropriate model sizes for different tasks
        4. Cache Results: Implement response caching to reduce redundant calls
        5. Batch Operations: Group similar requests to improve efficiency

        Cost optimization can reduce AI expenses by 40-60% while maintaining quality.
        """
        ),
        Document(
            text="""
        RAG (Retrieval-Augmented Generation) Pipeline Components and Costs:

        Embedding Generation:
        - Cost: ~$0.0001 per 1K tokens
        - Used for: Converting documents and queries to vectors
        - Optimization: Cache embeddings, use smaller models for simple content

        Vector Storage and Retrieval:
        - Cost: Varies by provider (Pinecone, Chroma, FAISS)
        - Used for: Storing and searching document embeddings
        - Optimization: Tune similarity thresholds, use hybrid search

        Response Synthesis:
        - Cost: $0.001-0.06 per 1K tokens (model dependent)
        - Used for: Generating final answers from retrieved context
        - Optimization: Use cheaper models for simple queries
        """
        ),
        Document(
            text="""
        Team-Based AI Governance Framework:

        Research Team: Budget $500/month
        - Focus on experimentation with advanced models
        - Higher cost tolerance for quality outcomes

        Engineering Team: Budget $200/month
        - Production workloads with cost efficiency focus
        - Prefer proven models with predictable costs

        Marketing Team: Budget $100/month
        - Content generation and customer support
        - Balance between quality and cost-effectiveness

        Governance ensures accountability and prevents budget overruns.
        """
        ),
    ]


def existing_rag_pipeline_without_genops(documents: list):
    """
    This represents a typical existing LlamaIndex RAG pipeline.
    NO GenOps code here - this is what users already have.
    """
    from llama_index.core import VectorStoreIndex

    print("📋 BEFORE: Running existing RAG pipeline (no tracking)...")

    # Build index
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=3)

    # Run some queries
    queries = [
        "What are the best practices for AI cost management?",
        "How much do different RAG components cost?",
        "What budget should each team have for AI?",
    ]

    responses = []
    for query in queries:
        print(f"   🤖 Query: {query[:50]}...")
        response = query_engine.query(query)
        responses.append(response.response)

    print("   ✅ Pipeline completed (but no cost visibility)")
    return responses


def existing_rag_pipeline_with_genops(documents: list):
    """
    The SAME pipeline but with GenOps auto-instrumentation enabled.
    Only difference is the auto_instrument() call at the top.
    """
    from llama_index.core import VectorStoreIndex

    from genops.providers.llamaindex import (
        auto_instrument,
        create_llamaindex_cost_context,
    )

    print("\n📊 AFTER: Same pipeline with GenOps auto-instrumentation...")

    # ONLY ADDITION: Enable auto-instrumentation
    auto_instrument()
    print("   ✅ GenOps auto-instrumentation enabled")

    # Use cost context for budget tracking
    with create_llamaindex_cost_context("team_demo", budget_limit=1.0) as cost_tracker:
        # IDENTICAL CODE - build index
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine(similarity_top_k=3)

        # IDENTICAL CODE - run queries
        queries = [
            "What are the best practices for AI cost management?",
            "How much do different RAG components cost?",
            "What budget should each team have for AI?",
        ]

        responses = []
        for i, query in enumerate(queries):
            print(f"   🤖 Query {i + 1}: {query[:50]}...")
            response = query_engine.query(query)
            responses.append(response.response)

        # NEW: Get comprehensive cost breakdown
        summary = cost_tracker.get_current_summary()

        print("   ✅ Pipeline completed WITH full cost visibility!")
        print("\n💰 COST BREAKDOWN:")
        print(f"   Total Cost: ${summary.total_cost:.6f}")
        print(f"   Operations: {summary.operation_count}")
        print(f"   Embedding Cost: ${summary.cost_breakdown.embedding_cost:.6f}")
        print(f"   Retrieval Cost: ${summary.cost_breakdown.retrieval_cost:.6f}")
        print(f"   Synthesis Cost: ${summary.cost_breakdown.synthesis_cost:.6f}")

        if summary.cost_breakdown.optimization_suggestions:
            print("\n💡 OPTIMIZATION SUGGESTIONS:")
            for suggestion in summary.cost_breakdown.optimization_suggestions:
                print(f"   • {suggestion}")

    return responses


def demonstrate_team_attribution():
    """Show how to add team/project attribution with minimal code changes."""
    from llama_index.core import Document, VectorStoreIndex

    from genops.providers.llamaindex import instrument_llamaindex

    print("\n🏷️ DEMO: Team Attribution (just add governance parameters)")

    # Create adapter with team defaults
    adapter = instrument_llamaindex(
        team="engineering", project="customer-support", environment="production"
    )

    # Sample support document
    support_doc = Document(
        text="""
    Customer Support FAQ:
    Q: How do I reset my password?
    A: Click 'Forgot Password' on the login page and follow the email instructions.

    Q: How do I upgrade my account?
    A: Visit Account Settings > Billing > Upgrade Plan to see available options.

    Q: Who do I contact for technical issues?
    A: Email support@company.com or use the live chat feature.
    """
    )

    index = VectorStoreIndex.from_documents([support_doc])
    query_engine = index.as_query_engine()

    # Track queries with team attribution
    customer_queries = [
        ("How do I reset my password?", "customer_onboarding"),
        ("What are the upgrade options?", "sales_support"),
        ("Who handles technical support?", "issue_resolution"),
    ]

    for query, feature in customer_queries:
        print(f"   📞 Customer Query: {query}")

        # Same query method, but with governance attributes
        response = adapter.track_query(
            query_engine,
            query,
            team="engineering",
            project="customer-support",
            feature=feature,
            customer_id="demo-customer-123",
        )

        print(f"   🤖 Response: {response.response[:80]}...")
        print(f"   🏷️ Attributed to: engineering/customer-support/{feature}")

    print("   ✅ All queries tracked with full team attribution!")


def main():
    """Main demonstration of auto-instrumentation capabilities."""
    print("🔧 GenOps LlamaIndex Auto-Instrumentation Demo")
    print("=" * 50)

    try:
        # Setup
        provider = setup_llm_provider()
        print(f"✅ LLM Provider: {provider}")

        documents = create_sample_documents()
        print(f"✅ Created {len(documents)} sample documents")

        # Demo 1: Before and After Comparison
        print("\n" + "=" * 50)
        print("DEMO 1: Zero-Code Transformation")
        print("=" * 50)

        existing_rag_pipeline_without_genops(documents)
        existing_rag_pipeline_with_genops(documents)

        # Demo 2: Team Attribution
        print("\n" + "=" * 50)
        print("DEMO 2: Team Attribution")
        print("=" * 50)

        demonstrate_team_attribution()

        # Summary
        print("\n" + "=" * 50)
        print("🎉 AUTO-INSTRUMENTATION COMPLETE!")
        print("=" * 50)

        print("✅ WHAT YOU ACCOMPLISHED:")
        print("   • Added GenOps tracking with ZERO code changes to existing RAG")
        print("   • Automatic cost breakdown (embedding, retrieval, synthesis)")
        print("   • Team and project attribution for governance")
        print("   • Budget monitoring with optimization suggestions")
        print("   • OpenTelemetry export to your observability platform")

        print("\n🎯 KEY INSIGHTS:")
        print("   • GenOps works with existing LlamaIndex code unchanged")
        print("   • Just add auto_instrument() at the top of your files")
        print("   • Comprehensive cost tracking across all RAG components")
        print("   • Team attribution enables department-level budgeting")
        print("   • Optimization suggestions help reduce costs automatically")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")

        if "api key" in str(e).lower():
            print("\n🔧 API KEY ISSUE:")
            print("   Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY")
        else:
            print("\n🔧 For detailed diagnostics run:")
            print(
                '   python -c "from genops.providers.llamaindex.validation import validate_setup, print_validation_result; print_validation_result(validate_setup(), detailed=True)"'
            )

        return False


if __name__ == "__main__":
    success = main()

    if success:
        print("\n🚀 READY FOR PHASE 3? (Production Deployment)")
        print("   → python production_rag_deployment.py     # Enterprise features")
        print("   → python advanced_agent_governance.py     # Agent workflows")
        print()
        print("📚 Or continue with Phase 2:")
        print("   → python rag_pipeline_tracking.py         # Detailed RAG monitoring")
    else:
        print("\n💡 Need help?")
        print("   → examples/llamaindex/README.md#troubleshooting")

    exit(0 if success else 1)
