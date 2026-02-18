"""
Example: RAG Pipeline Monitoring with GenOps
Demonstrates: Complete RAG workflow tracking including retrieval, embedding, and generation costs
Use case: Knowledge base applications, document Q&A systems, and retrieval-augmented generation
"""

import logging
import os

# Core LangChain imports
try:
    from langchain.chains import RetrievalQA
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.llms import OpenAI
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import Chroma
except ImportError:
    print("❌ LangChain not installed. Run: pip install langchain chromadb")
    exit(1)

# GenOps imports
try:
    from genops.providers.langchain import (
        create_chain_cost_context,
        instrument_langchain,
    )
    from genops.providers.langchain.rag_monitor import LangChainRAGInstrumentor
except ImportError:
    print("❌ GenOps not installed. Run: pip install genops-ai[langchain]")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_environment() -> bool:
    """Verify required environment variables."""
    required_vars = ["OPENAI_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        print(f"❌ Missing required variables: {missing}")
        return False

    # Set optional defaults
    if not os.getenv("OTEL_SERVICE_NAME"):
        os.environ["OTEL_SERVICE_NAME"] = "rag-pipeline-example"

    print("✅ Environment configured for RAG pipeline monitoring")
    return True


def create_sample_knowledge_base() -> list[Document]:
    """Create sample documents for the RAG knowledge base."""

    sample_documents = [
        {
            "content": """
            Artificial Intelligence (AI) Governance refers to the framework of policies, procedures,
            and practices designed to ensure responsible AI development and deployment. Key components
            include ethical guidelines, risk management, transparency requirements, and accountability
            mechanisms. Organizations implementing AI governance typically establish cross-functional
            committees, conduct regular audits, and maintain comprehensive documentation of AI systems.
            """,
            "metadata": {
                "source": "ai_governance_guide.pdf",
                "section": "introduction",
                "priority": "high",
            },
        },
        {
            "content": """
            Cost management in AI systems involves tracking computational expenses, model training costs,
            and inference pricing across different providers. Best practices include implementing
            cost monitoring dashboards, setting budget alerts, optimizing model selection based on
            cost-performance ratios, and establishing cost allocation frameworks for different business
            units or projects. OpenTelemetry integration enables real-time cost visibility.
            """,
            "metadata": {
                "source": "cost_management.pdf",
                "section": "best_practices",
                "priority": "medium",
            },
        },
        {
            "content": """
            Retrieval-Augmented Generation (RAG) combines the power of large language models with
            external knowledge retrieval. The process involves embedding documents into vector
            representations, storing them in vector databases, retrieving relevant context based
            on user queries, and generating responses that incorporate the retrieved information.
            RAG systems require careful tuning of retrieval parameters, embedding models, and
            generation strategies.
            """,
            "metadata": {
                "source": "rag_technical_guide.pdf",
                "section": "overview",
                "priority": "high",
            },
        },
        {
            "content": """
            Observability in AI applications encompasses logging, metrics, and tracing across the
            entire AI pipeline. Key metrics include model performance, latency, throughput, error
            rates, and resource utilization. Distributed tracing helps identify bottlenecks in
            complex AI workflows. Modern observability platforms support custom metrics for AI-specific
            concerns like model drift, prediction accuracy, and bias detection.
            """,
            "metadata": {
                "source": "observability_handbook.pdf",
                "section": "ai_metrics",
                "priority": "medium",
            },
        },
        {
            "content": """
            Policy enforcement in AI systems requires automated mechanisms to ensure compliance with
            organizational rules and regulatory requirements. This includes content filtering,
            access controls, usage quotas, and audit logging. Policy engines can integrate with
            AI workflows to provide real-time enforcement, while governance dashboards provide
            visibility into policy violations and compliance status.
            """,
            "metadata": {
                "source": "policy_enforcement.pdf",
                "section": "automation",
                "priority": "high",
            },
        },
    ]

    documents = []
    for doc_data in sample_documents:
        doc = Document(
            page_content=doc_data["content"].strip(), metadata=doc_data["metadata"]
        )
        documents.append(doc)

    print(f"📚 Created {len(documents)} sample documents for knowledge base")
    return documents


def setup_vector_store_with_monitoring(documents: list[Document]) -> tuple:
    """Set up vector store with GenOps monitoring."""
    print("🔄 Setting up vector store with monitoring...")

    # Initialize GenOps adapter and RAG instrumentor
    adapter = instrument_langchain()
    LangChainRAGInstrumentor(adapter)

    # Create embeddings with instrumentation
    print("   Creating embeddings model...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        chunk_size=1000,  # Optimize for cost
    )

    # Instrument embeddings for cost tracking
    instrumented_embeddings = adapter.instrument_embeddings(
        embeddings, team="knowledge-base", project="rag-demo"
    )

    # Split documents for better retrieval
    print("   Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, length_function=len
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"   Split into {len(split_docs)} chunks")

    # Create vector store with cost tracking
    print("   Creating vector store and computing embeddings...")
    with create_chain_cost_context("vector_store_setup") as cost_context:
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=instrumented_embeddings,
            persist_directory=None,  # Use in-memory for demo
        )

        # Record setup cost
        cost_context.record_generation_cost(0.002)  # Estimated setup overhead

    setup_summary = cost_context.get_final_summary()
    if setup_summary:
        print(f"   ✅ Vector store setup cost: ${setup_summary.total_cost:.4f}")
        print(f"   📊 Embedding tokens processed: {setup_summary.total_tokens_input:,}")

    return vectorstore, adapter


def demonstrate_basic_rag_query(vectorstore, adapter):
    """Demonstrate basic RAG query with monitoring."""
    print("\n🔍 Basic RAG Query with Monitoring")
    print("=" * 50)

    # Create retriever with instrumentation
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},  # Retrieve top 3 documents
    )

    # Instrument retriever for monitoring
    instrumented_retriever = adapter.instrument_retriever(
        retriever, team="qa-system", project="knowledge-retrieval"
    )

    # Test query
    query = "What are the key components of AI governance?"
    print(f"📝 Query: {query}")

    # Perform instrumented RAG query
    documents = adapter.instrument_rag_query(
        query=query,
        retriever=instrumented_retriever,
        team="qa-system",
        project="knowledge-retrieval",
        k=3,
    )

    print(f"✅ Retrieved {len(documents)} documents:")
    for i, doc in enumerate(documents, 1):
        print(f"   {i}. Source: {doc.metadata.get('source', 'unknown')}")
        print(f"      Content: {doc.page_content[:100]}...")
        print(f"      Priority: {doc.metadata.get('priority', 'unknown')}")

    return documents


def demonstrate_complete_rag_pipeline(vectorstore, adapter):
    """Demonstrate complete RAG pipeline with end-to-end monitoring."""
    print("\n🔗 Complete RAG Pipeline Monitoring")
    print("=" * 60)

    # Create QA chain
    print("🔄 Setting up RetrievalQA chain...")

    llm = OpenAI(
        temperature=0.2,  # Low temperature for factual responses
        max_tokens=300,
        model_name="gpt-3.5-turbo-instruct",
    )

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 4, "score_threshold": 0.3},
    )

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Stuff all retrieved docs into prompt
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
    )

    # Test queries with comprehensive monitoring
    test_queries = [
        {
            "query": "How can organizations implement effective AI cost management?",
            "customer_id": "enterprise_customer_001",
            "team": "cost-optimization",
        },
        {
            "query": "What is RAG and how does it work?",
            "customer_id": "tech_customer_002",
            "team": "technical-support",
        },
        {
            "query": "What are the key requirements for AI observability?",
            "customer_id": "platform_customer_003",
            "team": "platform-engineering",
        },
    ]

    pipeline_results = []

    for i, query_config in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query_config['query']}")
        print(f"👤 Customer: {query_config['customer_id']}")

        with create_chain_cost_context(f"rag_query_{i}") as cost_context:
            try:
                # Execute QA chain with monitoring
                result = adapter.instrument_chain_run(
                    qa_chain,
                    query=query_config["query"],
                    # Governance attributes
                    team=query_config["team"],
                    project="rag-qa-system",
                    customer_id=query_config["customer_id"],
                    environment="demo",
                    # RAG-specific attributes
                    retrieval_type="similarity_score_threshold",
                    k=4,
                    score_threshold=0.3,
                )

                print(f"✅ Answer: {result['result'][:200]}...")
                print(f"📚 Sources: {len(result['source_documents'])} documents used")

                pipeline_results.append(
                    {
                        "query": query_config["query"],
                        "customer_id": query_config["customer_id"],
                        "answer": result["result"],
                        "sources": len(result["source_documents"]),
                        "success": True,
                    }
                )

            except Exception as e:
                print(f"❌ Query failed: {e}")
                pipeline_results.append(
                    {
                        "query": query_config["query"],
                        "customer_id": query_config["customer_id"],
                        "success": False,
                        "error": str(e),
                    }
                )

        # Show cost information for this query
        query_summary = cost_context.get_final_summary()
        if query_summary:
            print(f"💰 Query cost: ${query_summary.total_cost:.4f}")
            print(f"⏱️  Query time: {query_summary.total_time:.2f}s")
            print(
                f"🔢 Tokens used: {query_summary.total_tokens_input + query_summary.total_tokens_output}"
            )

    return pipeline_results


def demonstrate_vector_search_performance(vectorstore, adapter):
    """Demonstrate vector search performance monitoring."""
    print("\n⚡ Vector Search Performance Monitoring")
    print("=" * 60)

    search_scenarios = [
        {"query": "AI governance best practices", "k": 5, "search_type": "similarity"},
        {
            "query": "cost optimization strategies",
            "k": 3,
            "search_type": "similarity_score_threshold",
            "score_threshold": 0.4,
        },
        {
            "query": "observability metrics and monitoring",
            "k": 2,
            "search_type": "mmr",  # Maximal Marginal Relevance
            "fetch_k": 10,
        },
    ]

    performance_results = []

    for i, scenario in enumerate(search_scenarios, 1):
        print(f"\n🔍 Search Scenario {i}: {scenario['query']}")
        print(f"   Parameters: k={scenario['k']}, type={scenario['search_type']}")

        try:
            # Perform instrumented vector search
            search_kwargs = {
                k: v for k, v in scenario.items() if k not in ["query", "search_type"]
            }

            results = adapter.instrument_vector_search(
                vector_store=vectorstore,
                query=scenario["query"],
                search_type=scenario["search_type"],
                team="search-optimization",
                project="vector-performance-test",
                **search_kwargs,
            )

            print(f"   ✅ Found {len(results)} results")
            for j, doc in enumerate(results[:2], 1):  # Show first 2 results
                source = doc.metadata.get("source", "unknown")
                priority = doc.metadata.get("priority", "unknown")
                print(f"      {j}. {source} (priority: {priority})")
                print(f"         {doc.page_content[:80]}...")

            performance_results.append(
                {
                    "scenario": i,
                    "query": scenario["query"],
                    "results_count": len(results),
                    "success": True,
                }
            )

        except Exception as e:
            print(f"   ❌ Search failed: {e}")
            performance_results.append(
                {
                    "scenario": i,
                    "query": scenario["query"],
                    "success": False,
                    "error": str(e),
                }
            )

    return performance_results


def generate_rag_monitoring_report(pipeline_results, performance_results):
    """Generate comprehensive RAG monitoring report."""
    print("\n📊 RAG Pipeline Monitoring Report")
    print("=" * 70)

    # Pipeline success rate
    successful_queries = sum(1 for r in pipeline_results if r["success"])
    total_queries = len(pipeline_results)
    success_rate = (
        (successful_queries / total_queries * 100) if total_queries > 0 else 0
    )

    print("🎯 Pipeline Performance:")
    print(
        f"   ✅ Successful queries: {successful_queries}/{total_queries} ({success_rate:.1f}%)"
    )

    if successful_queries > 0:
        avg_sources = (
            sum(r.get("sources", 0) for r in pipeline_results if r["success"])
            / successful_queries
        )
        print(f"   📚 Average sources per query: {avg_sources:.1f}")

    # Search performance
    successful_searches = sum(1 for r in performance_results if r["success"])
    total_searches = len(performance_results)
    search_success_rate = (
        (successful_searches / total_searches * 100) if total_searches > 0 else 0
    )

    print("\n🔍 Search Performance:")
    print(
        f"   ✅ Successful searches: {successful_searches}/{total_searches} ({search_success_rate:.1f}%)"
    )

    if successful_searches > 0:
        avg_results = (
            sum(r.get("results_count", 0) for r in performance_results if r["success"])
            / successful_searches
        )
        print(f"   📄 Average results per search: {avg_results:.1f}")

    print("\n📈 Monitoring Capabilities Demonstrated:")
    print("   ✅ End-to-end RAG pipeline cost tracking")
    print("   ✅ Customer-specific cost attribution")
    print("   ✅ Embedding model usage monitoring")
    print("   ✅ Vector search performance metrics")
    print("   ✅ Retrieval quality and relevance tracking")
    print("   ✅ Generation cost and token usage")
    print("   ✅ Real-time performance monitoring")

    print("\n🎯 Business Value:")
    print("   💰 Complete cost visibility for RAG operations")
    print("   👥 Per-customer cost attribution for billing")
    print("   📊 Performance metrics for optimization")
    print("   🔍 Quality metrics for continuous improvement")
    print("   ⚠️  Error tracking and alerting")


def main():
    """Main RAG pipeline monitoring demonstration."""
    print("🚀 GenOps LangChain RAG Pipeline Monitoring")
    print("=" * 80)

    if not setup_environment():
        return

    try:
        # Create knowledge base
        print("📚 Creating sample knowledge base...")
        documents = create_sample_knowledge_base()

        # Set up vector store with monitoring
        vectorstore, adapter = setup_vector_store_with_monitoring(documents)

        # Demonstrate basic RAG query
        demonstrate_basic_rag_query(vectorstore, adapter)

        # Demonstrate complete RAG pipeline
        pipeline_results = demonstrate_complete_rag_pipeline(vectorstore, adapter)

        # Demonstrate vector search performance
        performance_results = demonstrate_vector_search_performance(
            vectorstore, adapter
        )

        # Generate comprehensive report
        generate_rag_monitoring_report(pipeline_results, performance_results)

        print("\n🎉 RAG pipeline monitoring demo completed successfully!")

        print("\n🎯 Next Steps:")
        print("   - Check your observability dashboard for detailed RAG telemetry")
        print("   - Try different embedding models and compare costs")
        print("   - Experiment with retrieval parameters and monitor impact")
        print("   - Set up cost alerts for high-volume RAG applications")
        print("   - Explore agent_decision_tracking.py for agent workflows")

    except Exception as e:
        print(f"\n❌ RAG monitoring demo failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("   - Ensure OpenAI API key is set: export OPENAI_API_KEY=your_key")
        print("   - Install chromadb: pip install chromadb")
        print("   - Verify GenOps installation: pip install genops-ai[langchain]")
        logger.exception("RAG monitoring demo error")


if __name__ == "__main__":
    main()
