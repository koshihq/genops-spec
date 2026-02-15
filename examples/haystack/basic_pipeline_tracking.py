#!/usr/bin/env python3
"""
Haystack AI Basic Pipeline Tracking with GenOps Governance

Demonstrates basic Haystack pipeline execution with automatic cost tracking and governance.
Perfect starting point for integrating Haystack AI with GenOps governance controls.

Usage:
    python basic_pipeline_tracking.py

Features:
    - Simple pipeline creation with OpenAI generator
    - Automatic governance attribute collection
    - Component-level cost tracking and performance monitoring
    - Budget awareness and cost alerts
    - Pipeline execution metrics and insights
"""

import logging
import sys
from decimal import Decimal

# Core Haystack imports
try:
    from haystack import Document, Pipeline
    from haystack.components.builders import PromptBuilder
    from haystack.components.embedders import OpenAITextEmbedder  # noqa: F401
    from haystack.components.generators import OpenAIGenerator
    from haystack.components.retrievers import InMemoryBM25Retriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore
except ImportError as e:
    print(f"❌ Haystack not installed: {e}")
    print("Please install Haystack: pip install haystack-ai")
    sys.exit(1)

# GenOps imports
try:
    from genops.providers.haystack import (
        GenOpsHaystackAdapter,
        analyze_pipeline_costs,
        auto_instrument,
        validate_haystack_setup,
    )
except ImportError as e:
    print(f"❌ GenOps not installed: {e}")
    print("Please install GenOps: pip install genops-ai[haystack]")
    sys.exit(1)

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_environment() -> bool:
    """Verify required environment variables are set."""
    result = validate_haystack_setup()

    if result["is_valid"]:
        print("✅ Environment setup validated")
        print(f"Available providers: {result['available_providers']}")
        return True
    else:
        print("❌ Environment setup issues:")
        for issue in result["issues"]:
            print(f"  • {issue}")
        print("\nPlease set your API keys:")
        print("  export OPENAI_API_KEY='your-key-here'")
        return False


def create_simple_qa_pipeline() -> Pipeline:
    """Create a simple Q&A pipeline for demonstration."""
    print("\n🏗️ Creating Simple Q&A Pipeline")

    # Create pipeline
    pipeline = Pipeline()

    # Add components
    pipeline.add_component(
        "prompt_builder",
        PromptBuilder(
            template="""
        Answer the following question clearly and concisely:

        Question: {{question}}

        Answer:
        """
        ),
    )

    pipeline.add_component(
        "llm",
        OpenAIGenerator(
            model="gpt-3.5-turbo",
            generation_kwargs={"max_tokens": 150, "temperature": 0.7},
        ),
    )

    # Connect components
    pipeline.connect("prompt_builder", "llm")

    print("✅ Pipeline created with components: prompt_builder -> llm")
    return pipeline


def create_rag_pipeline() -> Pipeline:
    """Create a simple RAG pipeline for demonstration."""
    print("\n🏗️ Creating Simple RAG Pipeline")

    # Create document store with sample documents
    document_store = InMemoryDocumentStore()

    # Add sample documents about AI and machine learning
    documents = [
        Document(
            content="Artificial Intelligence (AI) is the simulation of human intelligence in machines. It includes machine learning, natural language processing, and computer vision."
        ),
        Document(
            content="Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed."
        ),
        Document(
            content="Retrieval-Augmented Generation (RAG) combines information retrieval with text generation to produce more accurate and contextual responses."
        ),
        Document(
            content="OpenAI's GPT models are transformer-based language models trained on large amounts of text data to generate human-like text."
        ),
        Document(
            content="Haystack is an open-source framework for building AI applications with components for document processing, retrieval, and generation."
        ),
    ]

    document_store.write_documents(documents)

    # Create pipeline
    pipeline = Pipeline()

    # Add components
    pipeline.add_component(
        "retriever", InMemoryBM25Retriever(document_store=document_store, top_k=2)
    )

    pipeline.add_component(
        "prompt_builder",
        PromptBuilder(
            template="""
        Use the following context to answer the question:

        Context:
        {% for document in documents %}
        {{ document.content }}
        {% endfor %}

        Question: {{question}}

        Answer based on the context:
        """
        ),
    )

    pipeline.add_component(
        "llm",
        OpenAIGenerator(
            model="gpt-3.5-turbo",
            generation_kwargs={"max_tokens": 200, "temperature": 0.5},
        ),
    )

    # Connect components
    pipeline.connect("retriever", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")

    print("✅ RAG pipeline created with components: retriever -> prompt_builder -> llm")
    return pipeline


def demo_auto_instrumentation():
    """Demonstrate zero-code auto-instrumentation."""
    print("\n" + "=" * 60)
    print("🚀 Demo 1: Zero-Code Auto-Instrumentation")
    print("=" * 60)

    # Enable auto-instrumentation
    print("Enabling auto-instrumentation...")
    success = auto_instrument(
        team="demo-team",
        project="basic-tracking",
        daily_budget_limit=20.0,
        governance_policy="advisory",
    )

    if not success:
        print("❌ Failed to enable auto-instrumentation")
        return

    print("✅ Auto-instrumentation enabled")

    # Create and run simple pipeline
    pipeline = create_simple_qa_pipeline()

    print("\n🔥 Running pipeline with auto-instrumentation...")
    result = pipeline.run(
        {
            "prompt_builder": {
                "question": "What are the main benefits of using Haystack AI for building AI applications?"
            }
        }
    )

    print("🎯 Pipeline Response:")
    print(f"   {result['llm']['replies'][0]}")

    # Get cost summary from auto-instrumentation
    from genops.providers.haystack import get_cost_summary, get_execution_metrics

    cost_summary = get_cost_summary()
    if "error" not in cost_summary:
        print("\n📊 Auto-Instrumentation Metrics:")
        print(f"   Daily costs: ${cost_summary['daily_costs']:.6f}")
        print(f"   Budget utilization: {cost_summary['daily_budget_utilization']:.1f}%")

    execution_metrics = get_execution_metrics()
    if "error" not in execution_metrics:
        print(f"   Total executions: {execution_metrics.get('total_executions', 0)}")


def demo_manual_governance():
    """Demonstrate manual governance with full control."""
    print("\n" + "=" * 60)
    print("🎛️ Demo 2: Manual Governance Control")
    print("=" * 60)

    # Create adapter with governance settings
    adapter = GenOpsHaystackAdapter(
        team="manual-demo",
        project="pipeline-tracking",
        environment="development",
        daily_budget_limit=15.0,
        monthly_budget_limit=300.0,
        governance_policy="advisory",
        enable_cost_alerts=True,
    )

    print("✅ GenOps Haystack adapter created")
    print(f"   Team: {adapter.team}")
    print(f"   Project: {adapter.project}")
    print(f"   Daily budget: ${adapter.daily_budget_limit}")

    # Create pipelines
    qa_pipeline = create_simple_qa_pipeline()
    rag_pipeline = create_rag_pipeline()

    # Track Q&A pipeline execution
    print("\n🔥 Tracking Q&A Pipeline Execution...")
    with adapter.track_pipeline("simple-qa", customer_id="demo-customer") as context:
        result = qa_pipeline.run(
            {
                "prompt_builder": {
                    "question": "How does Haystack AI help developers build better AI applications?"
                }
            }
        )

        print("🎯 Q&A Response:")
        print(f"   {result['llm']['replies'][0]}")

    # Get Q&A pipeline metrics
    qa_metrics = context.get_metrics()
    print("\n📊 Q&A Pipeline Metrics:")
    print(f"   Total cost: ${qa_metrics.total_cost:.6f}")
    print(f"   Components: {qa_metrics.total_components}")
    print(f"   Execution time: {qa_metrics.total_execution_time_seconds:.2f}s")
    print(f"   Cost by provider: {qa_metrics.cost_by_provider}")

    # Track RAG pipeline execution
    print("\n🔥 Tracking RAG Pipeline Execution...")
    with adapter.track_pipeline("simple-rag", use_case="document-qa") as context:
        result = rag_pipeline.run(
            {
                "retriever": {"query": "What is Retrieval-Augmented Generation?"},
                "prompt_builder": {
                    "question": "What is Retrieval-Augmented Generation?"
                },
            }
        )

        print("🎯 RAG Response:")
        print(f"   {result['llm']['replies'][0]}")

    # Get RAG pipeline metrics
    rag_metrics = context.get_metrics()
    print("\n📊 RAG Pipeline Metrics:")
    print(f"   Total cost: ${rag_metrics.total_cost:.6f}")
    print(f"   Components: {rag_metrics.total_components}")
    print(f"   Execution time: {rag_metrics.total_execution_time_seconds:.2f}s")
    print(f"   Most expensive component: {rag_metrics.most_expensive_component}")

    return adapter


def demo_session_tracking(adapter: GenOpsHaystackAdapter):
    """Demonstrate session-based tracking across multiple pipelines."""
    print("\n" + "=" * 60)
    print("📋 Demo 3: Session-Based Multi-Pipeline Tracking")
    print("=" * 60)

    # Create pipelines
    qa_pipeline = create_simple_qa_pipeline()

    # Track session with multiple pipeline executions
    with adapter.track_session(
        "comprehensive-demo",
        customer_id="demo-customer",
        use_case="pipeline-comparison",
    ) as session:
        print(f"📋 Started session: {session.session_name}")
        print(f"   Session ID: {session.session_id}")

        # Run multiple Q&A operations with different questions
        questions = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What are the benefits of using AI frameworks like Haystack?",
            "How can developers get started with building AI applications?",
        ]

        session_results = []
        for i, question in enumerate(questions, 1):
            print(f"\n   🔥 Pipeline execution {i}/{len(questions)}")

            with adapter.track_pipeline(f"qa-operation-{i}") as pipeline_ctx:
                result = qa_pipeline.run({"prompt_builder": {"question": question}})

                session_results.append(
                    {
                        "question": question,
                        "answer": result["llm"]["replies"][0][:100] + "...",
                        "cost": pipeline_ctx.get_metrics().total_cost
                        if pipeline_ctx.get_metrics()
                        else Decimal("0"),
                        "time": pipeline_ctx.get_metrics().total_execution_time_seconds
                        if pipeline_ctx.get_metrics()
                        else 0,
                    }
                )

                print(f"      Question: {question}")
                print(f"      Answer: {result['llm']['replies'][0][:80]}...")
                metrics = pipeline_ctx.get_metrics()
                if metrics:
                    print(f"      Cost: ${metrics.total_cost:.6f}")
                    print(f"      Time: {metrics.total_execution_time_seconds:.2f}s")

        print("\n📊 Session Summary:")
        print(f"   Total operations: {session.total_pipelines}")
        print(f"   Total cost: ${session.total_cost:.6f}")

        if session_results:
            avg_cost = sum(float(r["cost"]) for r in session_results) / len(
                session_results
            )
            avg_time = sum(r["time"] for r in session_results) / len(session_results)
            print(f"   Average cost per operation: ${avg_cost:.6f}")
            print(f"   Average execution time: {avg_time:.2f}s")


def demo_cost_analysis(adapter: GenOpsHaystackAdapter):
    """Demonstrate cost analysis and optimization recommendations."""
    print("\n" + "=" * 60)
    print("💰 Demo 4: Cost Analysis & Optimization")
    print("=" * 60)

    # Get comprehensive cost analysis
    analysis = analyze_pipeline_costs(adapter, time_period_hours=1)

    if "error" in analysis:
        print(f"❌ Cost analysis error: {analysis['error']}")
        return

    print("📊 Cost Analysis Results:")
    print(f"   Total cost (last hour): ${analysis['total_cost']:.6f}")

    if analysis["cost_by_provider"]:
        print("   Cost by provider:")
        for provider, cost in analysis["cost_by_provider"].items():
            print(f"     • {provider}: ${cost:.6f}")

    if analysis["cost_by_component"]:
        print("   Cost by component:")
        for component, cost in analysis["cost_by_component"].items():
            print(f"     • {component}: ${cost:.6f}")

    if analysis["most_expensive_component"]:
        print(f"   Most expensive component: {analysis['most_expensive_component']}")

    if analysis["recommendations"]:
        print("\n💡 Cost Optimization Recommendations:")
        for rec in analysis["recommendations"]:
            print(f"   • {rec['reasoning']}")
            print(f"     Current: {rec['current_provider']}")
            print(f"     Recommended: {rec['recommended_provider']}")
            print(f"     Potential savings: ${rec['potential_savings']:.6f}")
    else:
        print("\n✅ No cost optimization recommendations at this time")

    # Show overall cost summary
    cost_summary = adapter.get_cost_summary()
    print("\n📈 Overall Cost Summary:")
    print(f"   Daily spending: ${cost_summary['daily_costs']:.6f}")
    print(f"   Budget utilization: {cost_summary['daily_budget_utilization']:.1f}%")
    print(f"   Monthly projection: ${cost_summary['daily_costs'] * 30:.2f}")

    if cost_summary["daily_budget_utilization"] > 70:
        print("⚠️  High budget utilization - consider cost optimization")
    else:
        print("✅ Spending within comfortable limits")


def main():
    """Run the comprehensive Haystack basic tracking demonstration."""
    print("🏗️ Haystack AI Basic Pipeline Tracking with GenOps")
    print("=" * 60)

    # Setup and validate environment
    if not setup_environment():
        return 1

    try:
        # Demo 1: Auto-instrumentation
        demo_auto_instrumentation()

        # Demo 2: Manual governance
        adapter = demo_manual_governance()

        # Demo 3: Session tracking
        demo_session_tracking(adapter)

        # Demo 4: Cost analysis
        demo_cost_analysis(adapter)

        print("\n🎉 Basic pipeline tracking demonstration completed!")
        print("\n🚀 Next Steps:")
        print("   • Try rag_workflow_governance.py for RAG-specific tracking")
        print("   • Run agent_workflow_tracking.py for agent system monitoring")
        print(
            "   • Explore multi_provider_cost_aggregation.py for advanced cost analysis"
        )
        print("   • Build production-ready pipelines with comprehensive governance! 🏗️")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Try running setup_validation.py to check your configuration")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
