#!/usr/bin/env python3
"""
RAG Workflow Governance with GenOps and Haystack

Demonstrates specialized RAG (Retrieval-Augmented Generation) workflow tracking with
GenOps governance controls, including retrieval optimization, generation monitoring,
and comprehensive RAG-specific analytics.

Usage:
    python rag_workflow_governance.py

Features:
    - RAG-optimized GenOps adapter with specialized tracking
    - Document store setup with knowledge base documents
    - Retrieval phase monitoring with document relevance scoring
    - Generation phase tracking with prompt optimization
    - End-to-end RAG pipeline governance and cost analysis
    - RAG performance insights and optimization recommendations
"""

import logging
import os
import sys
from decimal import Decimal
from typing import List, Dict, Any

# Core Haystack imports for RAG workflow
try:
    from haystack import Pipeline, Document
    from haystack.components.retrievers import InMemoryBM25Retriever
    from haystack.components.builders import PromptBuilder
    from haystack.components.generators import OpenAIGenerator
    from haystack.components.embedders import OpenAITextEmbedder
    from haystack.document_stores.in_memory import InMemoryDocumentStore
except ImportError as e:
    print(f"❌ Haystack not installed: {e}")
    print("Please install Haystack: pip install haystack-ai")
    sys.exit(1)

# GenOps imports
try:
    from genops.providers.haystack import (
        create_rag_adapter,
        validate_haystack_setup,
        print_validation_result,
        get_rag_insights,
        analyze_pipeline_costs
    )
except ImportError as e:
    print(f"❌ GenOps not installed: {e}")
    print("Please install GenOps: pip install genops-ai[haystack]")
    sys.exit(1)

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_knowledge_base() -> InMemoryDocumentStore:
    """Create and populate knowledge base with AI and ML documents."""
    print("🗂️ Setting up Knowledge Base")
    
    document_store = InMemoryDocumentStore()
    
    # Sample knowledge base documents about AI/ML
    documents = [
        Document(
            content="Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation. It works by first retrieving relevant documents from a knowledge base, then using those documents as context to generate more accurate and informed responses. RAG helps reduce hallucinations and provides more factual, grounded answers.",
            meta={"category": "RAG", "source": "AI Research", "difficulty": "intermediate"}
        ),
        Document(
            content="Large Language Models (LLMs) are neural networks trained on vast amounts of text data. They can generate human-like text, answer questions, write code, and perform many language tasks. Popular LLMs include GPT-4, Claude, and LLaMA. However, they have limitations including hallucinations, knowledge cutoffs, and high computational costs.",
            meta={"category": "LLMs", "source": "AI Fundamentals", "difficulty": "beginner"}
        ),
        Document(
            content="Vector embeddings are dense numerical representations of text that capture semantic meaning. In RAG systems, documents and queries are converted to embeddings, allowing for semantic similarity search. This enables retrieval based on meaning rather than just keyword matching. Common embedding models include OpenAI's text-embedding-ada-002 and sentence transformers.",
            meta={"category": "Embeddings", "source": "ML Engineering", "difficulty": "intermediate"}
        ),
        Document(
            content="Prompt engineering is the practice of designing and optimizing prompts to get better responses from language models. Effective prompts are clear, specific, and provide appropriate context. Techniques include few-shot learning, chain-of-thought prompting, and system message optimization. Good prompt engineering can significantly improve model performance.",
            meta={"category": "Prompt Engineering", "source": "AI Engineering", "difficulty": "intermediate"}
        ),
        Document(
            content="Machine Learning Operations (MLOps) is the practice of deploying, monitoring, and maintaining ML models in production. It includes model versioning, automated testing, continuous integration/deployment, monitoring for data drift, and model performance tracking. MLOps ensures reliable and scalable ML systems.",
            meta={"category": "MLOps", "source": "ML Engineering", "difficulty": "advanced"}
        ),
        Document(
            content="Natural Language Processing (NLP) is a field of AI focused on helping computers understand, interpret, and generate human language. Key NLP tasks include text classification, named entity recognition, sentiment analysis, machine translation, and question answering. Modern NLP heavily relies on transformer architectures and large language models.",
            meta={"category": "NLP", "source": "AI Fundamentals", "difficulty": "beginner"}
        ),
        Document(
            content="Transformer architectures revolutionized NLP through the attention mechanism. The attention mechanism allows models to focus on relevant parts of the input when processing each token. This enables better handling of long sequences and parallel processing. Transformers form the basis of modern LLMs like GPT and BERT.",
            meta={"category": "Transformers", "source": "Deep Learning", "difficulty": "advanced"}
        ),
        Document(
            content="Fine-tuning is the process of adapting a pre-trained model to a specific task or domain by training it on task-specific data. This is more efficient than training from scratch and often achieves better performance. Common fine-tuning approaches include full fine-tuning, LoRA (Low-Rank Adaptation), and instruction tuning.",
            meta={"category": "Fine-tuning", "source": "ML Engineering", "difficulty": "intermediate"}
        )
    ]
    
    # Write documents to store
    document_store.write_documents(documents)
    
    print(f"✅ Knowledge base created with {len(documents)} documents")
    return document_store


def create_rag_pipeline(document_store: InMemoryDocumentStore) -> Pipeline:
    """Create comprehensive RAG pipeline with retrieval and generation components."""
    print("🏗️ Creating RAG Pipeline")
    
    pipeline = Pipeline()
    
    # Document retriever - finds relevant documents
    pipeline.add_component("retriever", InMemoryBM25Retriever(
        document_store=document_store,
        top_k=3  # Retrieve top 3 most relevant documents
    ))
    
    # Prompt builder - constructs context-aware prompts
    pipeline.add_component("prompt_builder", PromptBuilder(
        template="""
        Use the following context information to answer the question. Be accurate and cite the information from the context when possible.
        
        Context:
        {% for document in documents %}
        Source: {{document.meta.source}} ({{document.meta.category}})
        Content: {{document.content}}
        
        {% endfor %}
        
        Question: {{question}}
        
        Answer based on the context above:
        """
    ))
    
    # Language model generator - produces final answers
    pipeline.add_component("llm", OpenAIGenerator(
        model="gpt-3.5-turbo",
        generation_kwargs={
            "max_tokens": 250,
            "temperature": 0.3,  # Lower temperature for more factual responses
            "top_p": 0.9
        }
    ))
    
    # Connect pipeline components
    pipeline.connect("retriever", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")
    
    print("✅ RAG pipeline created with components: retriever -> prompt_builder -> llm")
    return pipeline


def demo_rag_governance():
    """Demonstrate comprehensive RAG workflow governance with GenOps tracking."""
    print("\n" + "="*70)
    print("🧠 RAG Workflow Governance with GenOps")
    print("="*70)
    
    # Create RAG-specialized adapter
    rag_adapter = create_rag_adapter(
        team="research-team",
        project="rag-knowledge-system",
        daily_budget_limit=50.0,
        enable_retrieval_tracking=True,
        enable_generation_tracking=True
    )
    
    print("✅ RAG-specialized GenOps adapter created")
    print(f"   Team: {rag_adapter.team}")
    print(f"   Project: {rag_adapter.project}")
    print(f"   Daily budget: ${rag_adapter.daily_budget_limit}")
    
    # Setup knowledge base and pipeline
    document_store = setup_knowledge_base()
    rag_pipeline = create_rag_pipeline(document_store)
    
    # Test questions covering different aspects of the knowledge base
    test_questions = [
        {
            "question": "What is Retrieval-Augmented Generation and how does it work?",
            "category": "RAG Fundamentals",
            "expected_complexity": "intermediate"
        },
        {
            "question": "How do vector embeddings help with semantic search in RAG systems?",
            "category": "Technical Details", 
            "expected_complexity": "intermediate"
        },
        {
            "question": "What are the main limitations of Large Language Models?",
            "category": "LLM Knowledge",
            "expected_complexity": "beginner"
        },
        {
            "question": "How does fine-tuning differ from training a model from scratch?",
            "category": "ML Engineering",
            "expected_complexity": "intermediate"
        },
        {
            "question": "What role does the attention mechanism play in transformer architectures?",
            "category": "Deep Learning",
            "expected_complexity": "advanced"
        }
    ]
    
    # Execute RAG queries with comprehensive tracking
    session_results = []
    
    with rag_adapter.track_session("rag-qa-session", use_case="knowledge-base-qa") as session:
        print(f"\n📋 Started RAG session: {session.session_name}")
        
        for i, test_case in enumerate(test_questions, 1):
            question = test_case["question"]
            category = test_case["category"]
            
            print(f"\n🔍 Query {i}/{len(test_questions)}: {category}")
            print(f"   Question: {question}")
            
            # Track individual RAG pipeline execution
            with rag_adapter.track_pipeline(
                "rag-qa-query",
                customer_id="demo-customer",
                query_category=category,
                expected_complexity=test_case["expected_complexity"]
            ) as context:
                
                # Execute RAG pipeline
                result = rag_pipeline.run({
                    "retriever": {"query": question},
                    "prompt_builder": {"question": question}
                })
                
                answer = result["llm"]["replies"][0]
                retrieved_docs = result["retriever"]["documents"]
                
                print(f"   📚 Retrieved {len(retrieved_docs)} documents")
                print(f"   🎯 Answer: {answer[:150]}...")
                
                # Get execution metrics
                metrics = context.get_metrics()
                print(f"   💰 Cost: ${metrics.total_cost:.6f}")
                print(f"   ⏱️ Time: {metrics.total_execution_time_seconds:.2f}s")
                
                # Store results for analysis
                session_results.append({
                    "question": question,
                    "category": category,
                    "answer": answer,
                    "docs_retrieved": len(retrieved_docs),
                    "cost": float(metrics.total_cost),
                    "time": metrics.total_execution_time_seconds,
                    "pipeline_id": context.pipeline_id
                })
            
            session.add_pipeline_result(context.get_metrics())
        
        print(f"\n📊 RAG Session Summary:")
        print(f"   Total queries: {session.total_pipelines}")
        print(f"   Total cost: ${session.total_cost:.6f}")
        print(f"   Average cost per query: ${session.total_cost / session.total_pipelines:.6f}")
    
    return rag_adapter, session_results


def analyze_rag_performance(rag_adapter, session_results):
    """Analyze RAG performance with specialized insights."""
    print("\n" + "="*70)
    print("📈 RAG Performance Analysis")
    print("="*70)
    
    # Get overall cost analysis
    cost_analysis = analyze_pipeline_costs(rag_adapter, time_period_hours=1)
    
    print("💰 Cost Analysis:")
    print(f"   Total cost: ${cost_analysis['total_cost']:.6f}")
    print(f"   Cost by provider: {cost_analysis['cost_by_provider']}")
    print(f"   Most expensive component: {cost_analysis['most_expensive_component']}")
    
    # RAG-specific performance metrics
    if session_results:
        total_docs_retrieved = sum(r["docs_retrieved"] for r in session_results)
        avg_docs_per_query = total_docs_retrieved / len(session_results)
        avg_response_time = sum(r["time"] for r in session_results) / len(session_results)
        
        print(f"\n🧠 RAG-Specific Metrics:")
        print(f"   Average documents per query: {avg_docs_per_query:.1f}")
        print(f"   Average response time: {avg_response_time:.2f}s")
        print(f"   Total documents processed: {total_docs_retrieved}")
        
        # Performance by query category
        category_performance = {}
        for result in session_results:
            cat = result["category"]
            if cat not in category_performance:
                category_performance[cat] = {"costs": [], "times": [], "docs": []}
            
            category_performance[cat]["costs"].append(result["cost"])
            category_performance[cat]["times"].append(result["time"])
            category_performance[cat]["docs"].append(result["docs_retrieved"])
        
        print(f"\n📊 Performance by Query Category:")
        for category, perf_data in category_performance.items():
            avg_cost = sum(perf_data["costs"]) / len(perf_data["costs"])
            avg_time = sum(perf_data["times"]) / len(perf_data["times"])
            avg_docs = sum(perf_data["docs"]) / len(perf_data["docs"])
            
            print(f"   {category}:")
            print(f"     Average cost: ${avg_cost:.6f}")
            print(f"     Average time: {avg_time:.2f}s")
            print(f"     Average docs retrieved: {avg_docs:.1f}")
    
    # Get RAG-specific insights for individual queries
    print(f"\n🔍 Detailed RAG Insights:")
    for i, result in enumerate(session_results[:3], 1):  # Show first 3 for brevity
        if hasattr(rag_adapter, 'monitor'):
            # This would work with full implementation
            print(f"   Query {i} ({result['category']}):")
            print(f"     Documents retrieved: {result['docs_retrieved']}")
            print(f"     Processing time: {result['time']:.2f}s")
            print(f"     Cost efficiency: ${result['cost']/result['docs_retrieved']:.6f} per document")
    
    # Optimization recommendations
    if cost_analysis.get('recommendations'):
        print(f"\n💡 RAG Optimization Recommendations:")
        for rec in cost_analysis['recommendations']:
            print(f"   • Component: {rec['component']}")
            print(f"     Reasoning: {rec['reasoning']}")
            print(f"     Potential savings: ${rec['potential_savings']:.6f}")
    else:
        print(f"\n✅ RAG workflow is well-optimized - no major recommendations")


def demo_advanced_rag_features(rag_adapter):
    """Demonstrate advanced RAG features and governance patterns."""
    print("\n" + "="*70)
    print("🚀 Advanced RAG Features")
    print("="*70)
    
    # Multi-turn conversation simulation
    print("🗣️ Multi-turn Conversation Tracking:")
    
    conversation_history = []
    follow_up_questions = [
        "What is RAG?",
        "How does it reduce hallucinations?", 
        "What are the main components needed?",
        "How can I optimize RAG performance?"
    ]
    
    with rag_adapter.track_session("multi-turn-conversation", use_case="conversational-rag") as session:
        for i, question in enumerate(follow_up_questions, 1):
            print(f"\n   Turn {i}: {question}")
            
            # Build context from conversation history
            context_prompt = ""
            if conversation_history:
                context_prompt = "\n\nPrevious conversation:\n" + "\n".join([
                    f"Q: {prev['question']}\nA: {prev['answer'][:100]}..."
                    for prev in conversation_history[-2:]  # Last 2 turns
                ])
            
            with rag_adapter.track_pipeline(
                f"conversation-turn-{i}",
                turn_number=i,
                has_context=len(conversation_history) > 0
            ) as context:
                
                # Create document store for this example (in real scenario, would reuse)
                temp_store = setup_knowledge_base()
                temp_pipeline = create_rag_pipeline(temp_store)
                
                result = temp_pipeline.run({
                    "retriever": {"query": question + context_prompt},
                    "prompt_builder": {"question": question + context_prompt}
                })
                
                answer = result["llm"]["replies"][0]
                print(f"        Answer: {answer[:120]}...")
                
                conversation_history.append({
                    "question": question,
                    "answer": answer,
                    "turn": i
                })
                
                metrics = context.get_metrics()
                print(f"        Cost: ${metrics.total_cost:.6f} | Time: {metrics.total_execution_time_seconds:.2f}s")
            
            session.add_pipeline_result(context.get_metrics())
        
        print(f"\n   Conversation Summary:")
        print(f"     Total turns: {session.total_pipelines}")
        print(f"     Total cost: ${session.total_cost:.6f}")
        print(f"     Average cost per turn: ${session.total_cost / session.total_pipelines:.6f}")
    
    # Batch processing demonstration
    print(f"\n📦 Batch RAG Processing:")
    
    batch_questions = [
        "What are the benefits of using transformers in NLP?",
        "How does MLOps improve model deployment?",
        "What is the difference between fine-tuning and prompt engineering?"
    ]
    
    with rag_adapter.track_session("batch-rag-processing", use_case="batch-qa") as batch_session:
        batch_results = []
        
        for i, question in enumerate(batch_questions, 1):
            with rag_adapter.track_pipeline(f"batch-query-{i}", batch_position=i) as context:
                temp_store = setup_knowledge_base()
                temp_pipeline = create_rag_pipeline(temp_store)
                
                result = temp_pipeline.run({
                    "retriever": {"query": question},
                    "prompt_builder": {"question": question}
                })
                
                batch_results.append({
                    "question": question,
                    "answer": result["llm"]["replies"][0],
                    "cost": float(context.get_metrics().total_cost)
                })
            
            batch_session.add_pipeline_result(context.get_metrics())
        
        print(f"   Processed {len(batch_questions)} questions in batch")
        print(f"   Total batch cost: ${batch_session.total_cost:.6f}")
        print(f"   Efficiency: ${batch_session.total_cost / len(batch_questions):.6f} per question")


def main():
    """Run the comprehensive RAG workflow governance demonstration."""
    print("🧠 RAG Workflow Governance with Haystack + GenOps")
    print("="*70)
    
    # Validate environment setup
    print("🔍 Validating setup...")
    result = validate_haystack_setup()
    
    if not result.is_valid:
        print("❌ Setup validation failed!")
        print_validation_result(result)
        return 1
    else:
        print("✅ Environment validated and ready")
    
    try:
        # Main RAG governance demonstration
        rag_adapter, session_results = demo_rag_governance()
        
        # Analyze RAG performance
        analyze_rag_performance(rag_adapter, session_results)
        
        # Advanced RAG features
        demo_advanced_rag_features(rag_adapter)
        
        print("\n🎉 RAG Workflow Governance demonstration completed!")
        print("\n🚀 Next Steps:")
        print("   • Try agent_workflow_tracking.py for agent system monitoring")
        print("   • Explore multi_provider_cost_aggregation.py for cost optimization")
        print("   • Run enterprise_governance_patterns.py for advanced features")
        print("   • Build your own RAG system with complete governance! 🧠")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Demonstration interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)
        print(f"\n❌ Demo failed: {e}")
        print("Try running the setup validation to check your configuration")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)